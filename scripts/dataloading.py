from scipy.interpolate.interpnd import (
    _ndim_coords_from_arrays as ndim_coords_from_arrays,
)
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.spatial import cKDTree as KDTree
from torch.utils.data import DataLoader, Dataset
import datetime as dt
import glob
import numpy as np
import os
import pandas as pd
import torch
import xarray as xr

from model import LABEL_FROZEN, LABEL_THAWED
from tb import (
    KEY_FREQ_POL,
    KEY_SAT_NAME,
    KEY_SAT_PASS,
    KEY_YEAR,
    KEY_37V,
    SAT_DESCENDING,
)
from validation_db_orm import (
    DbWMOMeanDate,
    DbWMOMetDailyTempRecord,
    DbWMOMetStation,
    date_to_int,
)
import ease_grid as eg
import tb as tbmod
import utils


class DataLoadingError(Exception):
    pass


class ViewCopyTransform:
    """A transform takes a view of the input and returns a copy"""

    def __init__(self, row_min, row_max, col_min, col_max):
        self.top = row_min
        self.bot = row_max + 1
        self.left = col_min
        self.right = col_max + 1

    def __call__(self, data):
        copy_func = np.copy if isinstance(data, np.ndarray) else torch.clone
        return copy_func(
            data[..., self.top : self.bot, self.left : self.right]
        )


class DatabaseReference:
    def __init__(self, path, creation_func):
        utils.validate_file_path(path)
        self.path = path
        self.creation_func = creation_func

    def __call__(self):
        return self.creation_func(self.path)


class GaussianRBF:
    """Gaussian radial basis function"""

    def __init__(self, epsilon):
        self.eps = epsilon

    def __call__(self, r):
        return np.exp(-((self.eps * r) ** 2))


DEFAULT_RBF_EPS = 8e-6


class AWSFuzzyLabelDataset(Dataset):
    def __init__(
        self,
        db_ref,
        rbf=GaussianRBF(DEFAULT_RBF_EPS),
        transform=None,
        other_mask=None,
        k=100,
        grid_code=eg.ML,
    ):
        assert k > 0, "k must be greater than 0"
        self.db_ref = db_ref
        self.rbf = rbf
        self.k = k
        self.grid_code = grid_code
        self.transform = transform or (lambda x: x)
        elon, elat = eg.v1_get_full_grid_lonlat(grid_code)
        ease_xm, ease_ym = eg.v1_lonlat_to_meters(elon, elat, grid_code)
        self.ease_xm = self.transform(ease_xm)
        self.ease_ym = self.transform(ease_ym)
        self.ease_tree = KDTree(
            list(zip(self.ease_xm.ravel(), self.ease_ym.ravel()))
        )
        elon = self.transform(elon)
        elat = self.transform(elat)
        # Compute bounds for querying the db
        self.lon_min = elon.min()
        self.lon_max = elon.max()
        self.lat_min = elat.min()
        self.lat_max = elat.max()
        self.grid_shape = self.ease_xm.shape
        other_mask = (
            other_mask
            if other_mask is not None
            else np.zeros(eg.GRID_NAME_TO_V1_SHAPE[eg.ML], dtype=bool)
        )
        self.other_mask = self.transform(other_mask)
        self.ft_mask = ~self.other_mask

    def __getitem__(self, dtime):
        if not isinstance(dtime, dt.datetime):
            raise TypeError("Index value must be a Datetime object")
        date = dt.date(dtime.year, dtime.month, dtime.day)
        hour = dtime.hour
        field = DbWMOMetDailyTempRecord.temperature_mean
        if hour == 6:
            field = DbWMOMetDailyTempRecord.temperature_min
        elif hour == 18:
            field = DbWMOMetDailyTempRecord.temperature_max
        records = (
            self.db_ref()
            .query(DbWMOMetStation.lon, DbWMOMetStation.lat, field)
            .join(DbWMOMetDailyTempRecord.met_station)
            .filter(DbWMOMetDailyTempRecord.date_int == date_to_int(date))
            .filter(field != None)  # noqa: E711  have to use != for sqlalchemy
            .filter(DbWMOMetStation.lon >= self.lon_min)
            .filter(DbWMOMetStation.lon <= self.lon_max)
            .filter(DbWMOMetStation.lat >= self.lat_min)
            .filter(DbWMOMetStation.lat <= self.lat_max)
            .all()
        )
        dgrid = np.full(np.prod(self.grid_shape), np.inf)
        fgrid = np.full(np.prod(self.grid_shape), np.inf)
        for i, r in enumerate(records):
            sx, sy = eg.v1_lonlat_to_meters(r[0], r[1])
            dist, idx = self.ease_tree.query((sx, sy), k=100)
            # P(frozen; x) := 0.5(P(frozen=True; x_stn)*RBF(x - x_stn))) + 0.5
            # P(thawed; x) := 1 - P(frozen; x)
            if r[-1] <= 273.15:
                # Frozen
                p_frozen = (0.5 * self.rbf(dist)) + 0.5
            else:
                # Thawed
                p_thawed = (0.5 * self.rbf(dist)) + 0.5
                p_frozen = 1 - p_thawed
            dist_min = dist < dgrid[idx]
            view = fgrid[idx]
            view[dist_min] = p_frozen[dist_min]
            fgrid[idx] = view
            view = dgrid[idx]
            view[dist_min] = dist[dist_min]
            dgrid[idx] = view
            # Handle tie
            dist_tie = dist == dgrid[idx]
            if dist_tie.any():
                view = fgrid[idx]
                view[dist_tie] = (p_frozen[dist_tie] + view[dist_tie]) / 2.0
                fgrid[idx] = view
                view = dgrid[idx]
                view[dist_tie] = dist[dist_tie]
                dgrid[idx] = view
        fgrid = fgrid.reshape(self.grid_shape)

        labels = np.zeros((3, *self.grid_shape))
        # Frozen
        labels[0] = fgrid
        # Thawed
        labels[1] = 1 - fgrid
        self.labels_copy = labels.copy()
        # fill with 50% everywhere else
        labels[0, np.isinf(fgrid)] = 0.5
        labels[1, np.isinf(fgrid)] = 0.5
        # OTHER: P(other) := 1 at mask points, 0 everywhere else
        labels[2, self.other_mask] = 1
        labels[2, self.ft_mask] = 0
        labels[0, self.other_mask] = 0
        labels[1, self.other_mask] = 0
        return labels

    def __len__(self):
        return self.db_ref().query(DbWMOMeanDate).count()


class AWSDateRangeWrapperDataset(Dataset):
    def __init__(self, aws_dataset, start_date, end_date, am_pm):
        """Make an AWS dataset that takes dates indexable with integers.

        start_date is inclusive while end_date is exclusive. am_pm can be "AM"
        or "PM".
        """
        assert (
            start_date < end_date
        ), "Start date must be less than the end date"
        hour = 6 if am_pm == "AM" else 18
        date = dt.datetime(
            start_date.year, start_date.month, start_date.day, hour
        )
        delta = dt.timedelta(days=1)
        dates = []
        while date < end_date:
            dates.append(date)
            date += delta
        dates = dates
        self.am_pm = am_pm
        self.idx_to_date = {i: d for i, d in enumerate(dates)}
        self.ds = aws_dataset

    def __getitem__(self, idx):
        return self.ds[self.idx_to_date[idx]]

    def __len__(self):
        return len(self.idx_to_date)


class ERA5BidailyDataset(Dataset):
    def __init__(
        self, paths, var_name, scheme, other_mask=None, transform=None
    ):
        self.ds = xr.open_mfdataset(paths, combine="by_coords")
        if var_name not in self.ds:
            raise KeyError(
                f"Variable name '{var_name}' is not present in specified data"
            )
        self.var_name = var_name
        self.data_slice = None
        if scheme == "AM":
            self.data_slice = np.s_[::2]
        elif scheme == "PM":
            self.data_slice = np.s_[1::2]
        else:
            raise ValueError(f"Unknown scheme option: '{scheme}'")
        self.length = len(self.data())
        self.lon = self.ds.lon.values
        self.lat = self.ds.lat.values
        elon, elat = eg.v1_get_full_grid_lonlat(eg.ML)
        self.elon = elon[0] + 180
        self.elat = elat[:, 0]
        self.transform = transform or (lambda x: x)
        self.n_classes = 2 + int(other_mask is not None)
        other_mask = (
            other_mask
            if other_mask is not None
            else np.zeros(eg.GRID_NAME_TO_V1_SHAPE[eg.ML], dtype=bool)
        )
        self.other_mask = self.transform(other_mask)
        self.grid_shape = self.other_mask.shape

    def data(self):
        return self.ds[self.var_name][self.data_slice]

    def __getitem__(self, idx):
        grid = self.data()[idx].values
        # Need to make lat increasing for interpolation
        ip = RBS(self.lat[::-1], self.lon, grid[::-1])
        igrid = ip(self.elat[::-1], self.elon)[::-1]
        # Roll along the lon dimension to center at lon=0
        igrid = np.roll(igrid, (self.elon.size // 2) + 1, axis=1)
        igrid = self.transform(igrid)
        out = np.zeros((self.n_classes, *self.grid_shape), dtype=int)
        # Frozen
        out[0] = igrid <= 273.15
        # Thawed
        out[1] = igrid > 273.15
        if self.n_classes > 2:
            # Other
            out[0:2, self.other_mask] = 0
            out[2, self.other_mask] = 1
        return out

    def __len__(self):
        return self.length


def _input_filter(fi):
    return (
        fi is not None
        and fi.grid_code == "ML"
        and fi.sat_pass == "D"
        and fi.freq_pol <= KEY_37V
    )


def _create_input_table(paths, filter_func=_input_filter):
    finfo = [tbmod.parse_nc_tb_fname(f) for f in paths]
    finfo = [fi for fi in finfo if filter_func(fi)]
    # List of frequency-polarization pairs
    fps = sorted(set(fi.freq_pol for fi in finfo))
    # Data in form:
    #    year sat_name grid_code sat_pass freq_pol                     path
    # 0  2009      F13        ML        A      19H  tb_2009_F13_ML_A_19H.nc
    # 1  2009      F13        ML        A      19V  tb_2009_F13_ML_A_19V.nc
    # 2  2009      F13        ML        A      22V  tb_2009_F13_ML_A_22V.nc
    # 3  2009      F13        ML        A      37H  tb_2009_F13_ML_A_37H.nc
    # 4  2009      F13        ML        A      37V  tb_2009_F13_ML_A_37V.nc
    # ...
    df = pd.DataFrame(finfo).sort_values(by=[KEY_YEAR, KEY_FREQ_POL])
    groups = list(df.groupby([KEY_YEAR, KEY_SAT_NAME, KEY_SAT_PASS]))
    rows = []
    # Partially transpose to form:
    #    year sat_name sat_pass                      19H  19V 22V 37H 37V ...
    # 0  2009      F13        A  tb_2009_F13_ML_A_19H.nc  ... ... ... ... ...
    # 1  2009      F17        A  tb_2009_F17_ML_A_19H.nc  ... ... ... ... ...
    # 2  2010      F17        A  tb_2010_F17_ML_A_19H.nc  ... ... ... ... ...
    # ...
    for k, di in groups:
        y, name, sp = k
        row = [y, name, sp]
        for r in di.itertuples():
            row.append(r.path)
        rows.append(row)
    cols = [KEY_YEAR, KEY_SAT_NAME, KEY_SAT_PASS, *fps]
    table = pd.DataFrame(rows, columns=cols)
    return table, fps


def _validate_table(table, freq_pols):
    for _, row in table.iterrows():
        files = row[freq_pols].values
        sizes = set()
        for f in files:
            n = xr.open_dataset(f).time.size
            sizes.add(n)
        if len(sizes) > 1:
            raise DataLoadingError(
                f"Data time dimensions did not match for inputs: {files}"
            )


class NCDataset(Dataset):
    def __init__(self, paths, var_name, transform=None):
        utils.validate_file_path_list(paths)
        self.ds = xr.open_mfdataset(paths, combine="by_coords")
        if var_name not in self.ds:
            raise KeyError(
                f"Variable name '{var_name}' is not present in specified data"
            )
        self.var_name = var_name
        self.transform = transform or (lambda x: x)

    def __getitem__(self, idx):
        return self.transform(self.ds[self.var_name][idx].values)

    def __len__(self):
        return len(self.ds[self.var_name])


class NCTbDataset(Dataset):
    def __init__(self, tb_dir, transform=None):
        utils.validate_dir_path(tb_dir)
        self.tb_dir = tb_dir
        files = glob.glob(os.path.join(self.tb_dir, "*.nc"))
        if len(files) == 0:
            raise DataLoadingError("Could not find data files")
        self.table, self.freq_pols = _create_input_table(files)
        _validate_table(self.table, self.freq_pols)
        # Use pass through as default transform
        self.transform = transform or (lambda x: x)

        # Cache of xarray.Dataset objects
        self._loaders = {}
        self.size = 0
        # Maps the input index to a row of data files in the data table
        self.index_to_table_row = {}
        # Maps the input index to the grid index within the files of a row
        self.index_to_row_inner_index = {}
        self._build_mapping()

    def _build_mapping(self):
        size = 0
        tab_mapping = {}
        inner_idx_mapping = {}
        for irow, row in self.table.iterrows():
            f = row[self.freq_pols[0]]
            n = xr.open_dataset(f).time.size
            for i in range(size, size + n):
                tab_mapping[i] = irow
                inner_idx_mapping[i] = i - size
            size += n
        self.size = size
        self.index_to_table_row = tab_mapping
        self.index_to_row_inner_index = inner_idx_mapping

    def _get_loaders(self, keys):
        loaders = [
            self._loaders[k] if k in self._loaders else xr.open_dataset(k)
            for k in keys
        ]
        for k, l in zip(keys, loaders):
            self._loaders[k] = l
        return loaders

    def _load_grids(self, idx):
        tabrow = self.table.iloc[self.index_to_table_row[idx]]
        files = tabrow[self.freq_pols].values
        loaders = self._get_loaders(files)
        inner_idx = self.index_to_row_inner_index[idx]
        grids = []
        loaded = [loader.tb[inner_idx].values for loader in loaders]
        for g in loaded:
            g[np.isnan(g)] = 0
        grids.extend(loaded)
        return self.transform(np.array(grids))

    def __getitem__(self, idx):
        if idx >= self.size or idx < 0:
            raise IndexError(f"Index {idx} out of range for size {self.size}")
        data = self._load_grids(idx)
        return data

    def __len__(self):
        return self.size


class NpyDataset(Dataset):
    """Wraps a .npy data file."""

    def __init__(self, data_file):
        self.data = np.load(data_file)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class RepeatDataset(Dataset):
    """Repeats a value(s) and presents a specified length."""

    def __init__(self, data, n):
        self.data = data
        self.n = n

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return self.n


class RangeSingularGridDataset(Dataset):
    """Generates single-value grids using values from a range object"""

    def __init__(self, range_obj, grid_shape):
        self.values = list(range_obj)
        self.shape = grid_shape

    def __getitem__(self, idx):
        return np.full(self.shape, self.values[idx])

    def __len__(self):
        return len(self.values)


class GridsStackDataset(Dataset):
    """Stacks dataset outputs into single tensor.

    This dataset essentially calls torch.cat([d[idx] for d in datasets], 0)
    to get an (N_channels, H, W) tensor.
    """

    def __init__(self, datasets):
        if not len(datasets):
            raise DataLoadingError("No datasets were provided")
        if len(set(len(d) for d in datasets)) > 1:
            raise DataLoadingError("Dataset sizes must match")
        self.datasets = datasets
        self.size = len(self.datasets[0])

    def __getitem__(self, idx):
        data = [d[idx] for d in self.datasets]
        shape_len = np.array([len(d.shape) for d in data], dtype=int).max()
        if shape_len <= 2:
            shape_len += 1
        unsqueezed = []
        for d in data:
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d)
            if len(d.shape) < shape_len:
                d = d.unsqueeze(0)
            unsqueezed.append(d)
        return torch.cat(unsqueezed, 0)

    def __len__(self):
        return self.size


class ComposedDataset(Dataset):
    def __init__(self, datasets):
        if not len(datasets):
            raise DataLoadingError("No datasets were provided")
        if len(set(len(d) for d in datasets)) > 1:
            raise DataLoadingError("Dataset sizes must match")
        self.datasets = datasets
        self.size = len(self.datasets[0])

    def __getitem__(self, idx):
        return [d[idx] for d in self.datasets]

    def __len__(self):
        return self.size


class ComposedDictDataset(Dataset):
    def __init__(self, datasets):
        if not len(datasets):
            raise DataLoadingError("No datasets were provided")
        if len(set(len(d) for d in datasets)) > 1:
            raise DataLoadingError("Dataset sizes must match")
        self.datasets = datasets
        self.size = len(self.datasets[0])

    def __getitem__(self, idx):
        if idx >= self.size or idx < 0:
            raise IndexError(f"{idx} is out of bounds")
        return {d.KEY: d[idx] for d in self.datasets}

    def __len__(self):
        return self.size


DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 6


def get_default_data_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=DEFAULT_NUM_WORKERS,
    )
