from scipy.interpolate.interpnd import (
    _ndim_coords_from_arrays as ndim_coords_from_arrays,
)
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
    date_to_int,
    DbWMOMetDailyTempRecord,
    DbWMOMetStation,
)
import easy_grid as eg
import tb as tbmod
import utils


class DataLoadingError(Exception):
    pass


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


class ViewCopyTransform:
    """A transform takes a view of the input and returns a copy"""

    def __init__(self, row_min, row_max, col_min, col_max):
        self.top = row_min
        self.bot = row_max + 1
        self.left = col_min
        self.right = col_max + 1

    def __call__(self, data):
        copy_func = np.copy if isinstance(np.ndarray, data) else torch.clone
        return copy_func(
            data[..., self.top : self.bottom, self.left : self.right]
        )


class ValidationDataGenerator:
    def __init__(self, db_connection, grid_code=eg.ML):
        self.dbc = db_connection
        self.grid_code = grid_code
        self.ease_xm, self.ease_ym = eg.v1_lonlat_to_meters(
            *eg.v1_get_full_grid_lonlat(grid_code), grid_code
        )

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
            self.dbc.query(DbWMOMetStation.lon, DbWMOMetStation.lat, field)
            .join(DbWMOMetDailyTempRecord.met_station)
            .filter(DbWMOMetDailyTempRecord.date_int == date_to_int(date))
            .filter(field != None)  # noqa: E711  have to use != for sqlalchemy
            .all()
        )
        vlon = [r[0] for r in records]
        v;lat = [r[1] for r in records]
        temp = np.array([r[2] for r in records])
        vft = np.empty_like(temp, dtype='uint8')
        vft[temp <= 273.15] = LABEL_FROZEN
        vft[temp > 273.15] = LABEL_THAWED
        vxm, vym = eg.v1_lonlat_to_meters(vlon, vlat, self.grid_code)
        vpoints = list(zip(vxm, vym))
        tree = KDTree(vpoints)
        vgrid = np.zeros(self.ease_xm.shape, dtype=int)
        xi = ndim_coords_from_arrays((self.ease_xm, self.ease_ym), ndim=2)
        dist, idx = tree.query(xi)
        vgrid[:] = vft[idx]
        return vgrid, dist


KEY_INPUT_DATA = "input_data"
KEY_TIME = "time"
KEY_VALIDATION_DATA = "validation_data"
KEY_DIST_DATA = "dist_data"


class NCTbDataset(Dataset):
    def __init__(self, root_data_dir, transform=None):
        utils.validate_dir_path(root_data_dir)
        self.data_root = root_data_dir
        tb_dir = os.path.join(self.data_root, "tb")
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

    def _load_input_grids_and_datetime(self, idx):
        tabrow = self.table.iloc[self.index_to_table_row[idx]]
        files = tabrow[self.freq_pols].values
        loaders = self._get_loaders(files)
        inner_idx = self.index_to_row_inner_index[idx]
        date = utils.datetime64_to_date(loaders[0].time[inner_idx].values)
        hour = 6 if tabrow[KEY_SAT_PASS] == SAT_DESCENDING else 18
        datetime = dt.datetime(
            date.year, date.month, date.day, hour, tzinfo=dt.timezone.utc
        )
        shape = loaders[0].tb.shape[1:]
        grids = []
        grids.append(np.full(shape, datetime.timestamp()))
        loaded = [loader.tb[inner_idx].values for loader in loaders]
        for g in loaded:
            g[np.isnan(g)] = 0
        grids.extend(loaded)
        return (
            self.transform(np.array(grids)),
            int(datetime.timestamp()),
        )

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range for size {self.size}")
        input_data, timestamp = self._load_input_grids_and_datetime(idx)
        # TODO: load validation data from db and create label
        return {KEY_INPUT_DATA: input_data, KEY_TIME: timestamp}

    def __len__(self):
        return self.size


class FTDataset(Dataset):
    def __init__(self, tb_dataset, validation_generator):
        self.tb = tb_dataset
        self.val_gen = validation_generator

    def __getitem__(self, idx):
        data_dict = self.tb[idx]
        time = data_dict[KEY_TIME]
        validation_grid, dist_grid = self.val_gen[
            dt.datetime.utcfromtimestamp(time)
        ]
        return {
            KEY_INPUT_DATA: data_dict[KEY_INPUT_DATA],
            KEY_VALIDATION_DATA: validation_grid,
            KEY_DIST_DATA: dist_grid,
        }

    def __len__(self):
        return len(self.tb)


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
