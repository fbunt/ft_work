import datetime
import glob
import gzip
import logging
import os
import sqlalchemy as sa
import tqdm

from validation_db_orm import (
    DbWMOMetStation,
    DbWMOMeanDate,
    DbWMOMetDailyTempRecord,
    get_db_session,
)


_INFO = logging.INFO
_WARN = logging.WARN
_ERROR = logging.ERROR
_CRITICAL = logging.CRITICAL
_EXCEPTION = "exception"

_LOG = {
    _INFO: logging.info,
    _WARN: logging.warning,
    _ERROR: logging.error,
    _CRITICAL: logging.critical,
    _EXCEPTION: logging.exception,
}


def log(lvl, msg, stdout=False):
    """Log a message using the logging module. If stdout is True, the message
    is also printed to the screen.
    """
    if lvl not in _LOG:
        logging.error("Invalid logging level used. MSG:'{msg}'")
        return

    if stdout:
        print(msg)
    _LOG[lvl](msg)


MISSING = -99999


def parse_sid_from_fname(fname):
    base = os.path.basename(fname)
    # Interpret as hex string to handle leading letter of some station IDs
    return int("".join(base.split("-")[:2]), 16)


def _tstamp_to_date(s):
    if not s:
        return None
    y = int(s[:4])
    m = int(s[4:6])
    d = int(s[6:8])
    return datetime.date(y, m, d)


def _fahrenheit_to_kelvin(t):
    return (5.0 / 9.0 * (t - 32.0)) + 273.15


def _parse_wmo_station_data_line(line):
    # Interpret as hex string to handle leading letter of some station IDs
    sid = int(line[:12].strip().replace(" ", ""), 16)
    d = int(line[14:22])
    t = float(line[24:30])
    n = int(line[31:33])
    if t != 9999.9:
        t = _fahrenheit_to_kelvin(t)
        max_ = float(line[102:108])
        max_ = _fahrenheit_to_kelvin(max_) if max_ == 9999.9 else None
        min_ = float(line[110:116])
        min_ = _fahrenheit_to_kelvin(min_) if min_ == 9999.9 else None
        return DbWMOMetDailyTempRecord(
            station_id=sid,
            date_int=d,
            nsamples=n,
            temperature_mean=t,
            temperature_min=min_,
            temperature_max=max_,
        )
    return None


def parse_wmo_station_data_file(fd):
    """Parse WMO met station data file records"""
    lines = []
    means = []
    for i, line in enumerate(fd):
        # Skip header line
        if i > 0:
            lines.append(line.decode("utf-8"))
    for i, line in enumerate(lines):
        try:
            mean = _parse_wmo_station_data_line(line)
            if mean:
                means.append(mean)
        except ValueError:
            log(_ERROR, f"Error reading '{fd.name}'", True)
            log(_ERROR, f"Line {i}: {repr(line)}", True)
    return means


def load_wmo_station_data(fname):
    """Load gzipped WMO met station data file into list of records"""
    with gzip.open(fname) as fd:
        return parse_wmo_station_data_file(fd)


def _parse_wmo_station_meta_line(line):
    # Interpret as hex string to handle leading letter of some station IDs
    sid = int(line[:12].strip().replace(" ", ""), 16)
    usaf = line[:6].strip()
    wban = line[7:12].strip()
    name = line[13:42].strip()
    country = line[43:45].strip()
    state = line[48:50].strip()
    try:
        lat = float(line[57:64])
        if lat == 0.0:
            lat = MISSING
    except ValueError:
        lat = MISSING
    try:
        lon = float(line[65:73])
        if lon == 0.0:
            lon = MISSING
    except ValueError:
        lon = MISSING
    try:
        z = float(line[74:81])
    except ValueError:
        z = MISSING
    dstart = _tstamp_to_date(line[82:90])
    dend = _tstamp_to_date(line[91:99])
    return DbWMOMetStation(
        station_id=sid,
        usaf_id_str=usaf,
        wban_id_str=wban,
        name=name,
        country=country,
        state=state,
        lon=lon,
        lat=lat,
        elevation=z,
        start_date=dstart,
        end_date=dend,
    )


def parse_wmo_station_meta_file(fd):
    """Parse the WMO met station table file"""
    for _ in range(22):
        # Skipp header
        fd.readline()
    stns = []
    for i, line in enumerate(fd):
        try:
            stns.append(_parse_wmo_station_meta_line(line))
        except ValueError as e:
            log(_ERROR, f"Failed to parse line: {i}", True)
            log(_EXCEPTION, e, True)
    return stns


def _validate_wmo_station(s):
    return s.lon != MISSING and s.lat != MISSING and s.station_id


def load_wmo_station_table(fd):
    """Load WMO met station table file into dict"""
    return {
        s.station_id: s
        for s in parse_wmo_station_meta_file(fd)
        if _validate_wmo_station(s)
    }


DUPLICATE_FILTER_KEY = "duplicate_filter"


def duplicate_filter(samples, sid, fname):
    """Remove duplicate records in the sample list.

    This is very, very rare but occasionally there will be duplicate samples
    with different means and sample counts for the same day.
    """
    dates = {s.date_int: [] for s in samples}
    for s in samples:
        dates[s.date_int].append(s)
    out = []
    for d in dates:
        if len(dates[d]) == 1:
            out.append(dates[d][0])
        else:
            log(_WARN, f"Duplicate:'{fname}':{dates[d][0]}")
    return [dates[d][0] for d in dates if len(dates[d]) == 1]


LOW_SAMPLE_COUNT_FILTER_KEY = "low_sample_count"


def low_sample_count_filter(samples, sid, fname, cutoff=2):
    """Remove samples with sample counts below the cutoff threshold.

    Wrap in a lambda func to use a non default cutoff.
    """
    count = len(samples)
    samples = [s for s in samples if s.nsamples >= cutoff]
    if len(samples) < count:
        log(_WARN, f"Sample count below threshold found in '{fname}'")
    return samples


INCORRECT_SID_FILTER_KEY = "invalid sid filter"


def incorrect_sid_filter(samples, sid, fname):
    """Remove samples with station IDs that don't match the ID of the file
    They were sourced from.
    """
    out = []
    for i, s in enumerate(samples):
        if s.station_id == sid:
            out.append(s)
        else:
            log(_WARN, f"Invalid sid: file: {fname}:{i+1}")
    return out


def _scrub_wmo_station_data(samples, sid, fname, filters):
    """Scrub the data for inconsistent state using supplied filters"""
    scrub_stats = {k: 0 for k in filters}
    # Apply filters
    for k, f in filters.items():
        count = len(samples)
        samples = f(samples, sid, fname)
        scrub_stats[k] = count - len(samples)
    return samples, scrub_stats


class WMOMetStationDataWrapper:
    """A class to wrap the WMO met station data set.

    Stations with missing meta data (lat/lon) or no data are dropped. Data
    record filtering is also applied. A table of met stations is supplied along
    with an easy interface to retrieve data records.
    """

    def __init__(
        self,
        files,
        stn_table,
        filters={
            INCORRECT_SID_FILTER_KEY: incorrect_sid_filter,
            DUPLICATE_FILTER_KEY: duplicate_filter,
            LOW_SAMPLE_COUNT_FILTER_KEY: low_sample_count_filter,
        },
    ):
        # Add to prevent invalid state
        if INCORRECT_SID_FILTER_KEY not in filters:
            filters[INCORRECT_SID_FILTER_KEY] = incorrect_sid_filter
        self.filters = filters
        fname_to_sid = {f: parse_sid_from_fname(f) for f in files}
        data_sids = set(sid for f, sid in fname_to_sid.items())
        # Get sids for stations with valid meta data and data in data files
        self.valid_sids = set(stn_table) & set(data_sids)

        self.files = sorted(
            [f for f, sid in fname_to_sid.items() if sid in self.valid_sids]
        )
        self.stn_table = {sid: stn_table[sid] for sid in self.valid_sids}
        self.idx_to_sid = {
            i: fname_to_sid[f] for i, f in enumerate(self.files)
        }
        self.filter_stats = {k: 0 for k in self.filters}
        self.filter_stats["skipped data files"] = len(files) - len(self.files)

        # Used to track new dates
        self.dates = set()
        # Track query indices to prevent duplication of stats
        self._queried_idxs = set()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Return dates and data records for the index specified.

        Data is filtered using the filters supplied at wrapper instantiantion.
        Dates are the new dates that have not been seen yet.
        """
        if idx < 0 or idx >= len(self.files):
            raise IndexError("Index out of bounds")
        sid = self.idx_to_sid[idx]
        samples, stats = _scrub_wmo_station_data(
            load_wmo_station_data(self.files[idx]),
            sid,
            self.files[idx],
            self.filters,
        )
        new_dates = []
        # Don't add stats if already computed. Don't recompute new dates unless
        # not queried yet
        if idx not in self._queried_idxs:
            for k in stats:
                self.filter_stats[k] += stats[k]
            dates = [DbWMOMeanDate(date_int=s.date_int) for s in samples]
            # Only return dates that haven't been encountered yet. set.add()
            # always returns None so use not
            new_dates = [
                d
                for d in dates
                if d.date not in self.dates and not self.dates.add(d.date)
            ]
        self._queried_idxs.add(idx)
        return new_dates, samples


def write_records_to_db(db, records, src, try_individual=True):
    """Write the supplied records to the db and return the number written
    successfully.

    This func attempts to add all of the records together and then falls back
    to adding them individually if an error occurs. Rollbacks are applied as
    needed.
    """
    written = 0
    try:
        db.add_all(records)
        db.commit()
        written += len(records)
    except sa.exc.IntegrityError as e:
        log(_ERROR, f"Error while adding '{src}'", True)
        log(_EXCEPTION, e)
        log(_INFO, "Adding individually", True)
        db.rollback()
        if try_individual:
            for r in records:
                try:
                    db.add(r)
                    db.commit()
                    written += 1
                except sa.exc.IntegrityError as e:
                    log(_ERROR, f"Error adding {r}")
                    log(_EXCEPTION, e)
                    db.rollback()
    return written


def add_wmo_data_to_db(db, data_wrapper):
    log(_INFO, "Adding WMO station table", True)
    stns_written = write_records_to_db(
        db, [s for sid, s in data_wrapper.stn_table.items()], "stn_table"
    )
    if stns_written < len(data_wrapper.stn_table):
        log(_ERROR, "Failed to add WMO station table", True)
        return

    log(_INFO, "Adding WMO data records", True)
    dates_written = 0
    data_records_written = 0
    # Use tqdm for nice progress bar and completion estimate
    for i in tqdm.tqdm(range(len(data_wrapper)), ncols=80):
        f = data_wrapper.files[i]
        logging.info(f"Adding '{f}' to db")
        new_dates, data = data_wrapper[i]
        if new_dates:
            dates_written += write_records_to_db(db, new_dates, f)
        data_records_written += write_records_to_db(db, data, f)
    log(_INFO, f"Station records written: {stns_written}", True)
    log(_INFO, f"Date records written: {dates_written}", True)
    log(_INFO, f"Data records written: {data_records_written}", True)
    log(_INFO, "Filter Stats:", True)
    for k, v in data_wrapper.filter_stats.items():
        log(_INFO, f"{k}: {v}", True)


def build_wmo_database(
    db_path, root_data_dir, stn_table_path, overwrite=False
):
    db_exists = os.path.isfile(db_path)
    if db_exists and not overwrite:
        raise IOError("Database file already exists")
    if not os.path.isdir(root_data_dir):
        raise IOError("Invalid root data path")
    if not os.path.isfile(stn_table_path):
        raise IOError("Invalid station table path")

    if db_exists:
        log(_INFO, "Removing old db file", True)
        os.remove(db_path)
    log(_INFO, f"Opening: '{db_path}'", True)
    db = get_db_session(db_path)

    log(_INFO, f"Loading met station table: '{stn_table_path}'", True)
    with open(stn_table_path) as fd:
        stns = load_wmo_station_table(fd)
    files = glob.glob(os.path.join(root_data_dir, "*/*.gz"))
    data_wrapper = WMOMetStationDataWrapper(files, stns)
    add_wmo_data_to_db(db, data_wrapper)
    db.commit()
    db.close()


def _validate_file(path):
    if os.path.isfile(path):
        return path
    raise IOError(f"Could not find file: {path}")


def _validate_directory(path):
    if os.path.isdir(path):
        return path
    raise IOError(f"Could not find directory: {path}")


def _get_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Create a database from WMO station records."
    )
    p.add_argument("db_path", type=str, help="Path of db file to create")
    p.add_argument(
        "root_data_dir", type=_validate_directory, help="Path to root data dir"
    )
    p.add_argument(
        "station_table_path",
        type=_validate_file,
        help="Path to WMO met station table",
    )
    p.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="Overwrite existing database file",
    )
    p.add_argument(
        "-l",
        "--logfile",
        type=str,
        default="db.log",
        help="Path of desired log file. DEFAULT: db.log",
    )
    return p


if __name__ == "__main__":
    args = _get_parser().parse_args()
    logging.basicConfig(
        filename=args.logfile,
        level=logging.DEBUG,
        filemode="w",
        format="%(asctime)s:%(levelname)s:%(message)s",
    )
    db_exists = os.path.isfile(args.db_path)
    if db_exists and not args.overwrite:
        raise IOError("Database file already exists. Use -O to overwrite")

    build_wmo_database(
        args.db_path,
        args.root_data_dir,
        args.station_table_path,
        args.overwrite,
    )
