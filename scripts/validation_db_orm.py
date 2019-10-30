from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
import datetime
import os
import sqlalchemy as sa


# Database base class for the schema
DbBase = declarative_base()


_STATION_USAF_SIZE = 6
_STATION_WBAN_SIZE = 5
_STATION_NAME_SIZE = 30
_STATION_COUNTRY_SIZE = 2
_STATION_STATE_SIZE = 2


class DbWMOMetStation(DbBase):
    """Class representing a WMO met station."""

    __tablename__ = "wmo_met_stations"

    id = Column(Integer, primary_key=True)

    # Int version of usaf id
    station_id = Column(
        "station_id", Integer, nullable=False, unique=True, index=True
    )
    # USAF ID string (hex)
    usaf_id_str = Column("usaf_id_str", String(_STATION_USAF_SIZE))
    # WBAN ID string
    wban_id_str = Column("wban_id", String(_STATION_WBAN_SIZE))
    # Station name string
    name = Column("name", String(_STATION_NAME_SIZE))
    # Country code
    country = Column("country", String(_STATION_COUNTRY_SIZE))
    # US State code, if applicable
    state = Column("state", String(_STATION_STATE_SIZE))
    lon = Column("lon", Float)
    lat = Column("lat", Float)
    # meters
    elevation = Column("elevation", Float)
    # UTC
    start_date = Column("start_date", DateTime)
    end_date = Column("end_date", DateTime)

    temperature_means = relationship(
        "DbWMOMetDailyTempMean",
        back_populates="met_station",
    )

    def __repr__(self):
        return (
            "<DbWMOMetStation("
            "id={0.station_id}, "
            "name={0.name}, "
            "lon={0.lon}, "
            "lat={0.lat}, "
            "elevation={0.elevation})>"
        ).format(self)


class DbWMOMeanDate(DbBase):
    """Class representing a sample date"""

    __tablename__ = "wmo_mean_dates"

    id = Column(Integer, primary_key=True, nullable=False, unique=True)

    date_int = Column(
        "date_int", Integer, nullable=False, unique=True, index=True
    )
    date = Column("date", DateTime, nullable=False, unique=True, index=True)

    temperature_means = relationship(
        "DbWMOMetDailyTempMean", back_populates="date"
    )

    def __init__(self, **kwargs):
        if "date" in kwargs and "date_int" in kwargs:
            self.date = kwargs["date"]
            self.date_int = kwargs["date_int"]
        if "date" in kwargs and "date_int" not in kwargs:
            self.date = kwargs["date"]
            self.date_int = date_to_int(self.date)
        if "date" not in kwargs and "date_int" in kwargs:
            self.date_int = kwargs["date_int"]
            self.date = int_to_date(self.date_int)

    def __repr__(self):
        return "<DbWMOMeanDate(date={0.date})>".format(self)


class DbWMOMetDailyTempMean(DbBase):
    """Class representing a single met station daily temperature mean"""

    __tablename__ = "wmo_met_daily_mean_data"

    id = Column(Integer, primary_key=True)

    # ID of station that took the samples for the mean
    station_id = Column(
        "station_id",
        Integer,
        ForeignKey(DbWMOMetStation.station_id),
        nullable=False,
        index=True,
    )
    # Date of sample as int
    date_int = Column(
        "date_int",
        Integer,
        ForeignKey(DbWMOMeanDate.date_int),
        nullable=False,
        index=True,
    )
    # The number of hourly samples used for the mean.
    nsamples = Column("nsamples", Integer)
    # Kelvin temperature
    temperature = Column("temperature", Float)

    met_station = relationship(
        "DbWMOMetStation", back_populates="temperature_means"
    )
    date = relationship("DbWMOMeanDate", back_populates="temperature_means")

    def __repr__(self):
        return (
            "<DbWMOMetDailyTempMean("
            "station_id={0.station_id}, "
            "date_int={0.date_int}, "
            "nsamples={0.nsamples}, "
            "temperature={0.temperature})>"
        ).format(self)


def int_to_date(i):
    """Convert int of the form 20180301 to a date object."""
    y = i // 10000
    m = (i // 100) - (y * 100)
    d = i - (y * 10000) - (m * 100)
    return datetime.date(y, m, d)


def date_to_int(date):
    """Convert a date object to an int of the form 20180301."""
    return (date.year * 10000) + (date.month * 100) + date.day


ENGINE_SQLITE = "sqlite"
ENGINE_POSTGRESQL = "postgresql"

ENGINE_TO_PREFIX = {
    ENGINE_SQLITE: "sqlite:///",
    ENGINE_POSTGRESQL: "postgresql:///",
}


def get_db_session(db_path, engine=ENGINE_SQLITE, base=DbBase):
    db_name = ENGINE_TO_PREFIX[engine] + os.path.abspath(db_path)
    engine_inst = sa.create_engine(db_name)
    base.metadata.create_all(engine_inst)
    db_impl = sessionmaker(bind=engine_inst)
    return db_impl()
