import cartopy.crs as ccrs
import datetime as dt
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from validation_db_orm import (
    DbWMOMetDailyTempMean,
    date_to_int,
    get_db_session,
)
import ease_grid as eg

date = dt.date(2000, 1, 1)
db = get_db_session("../data/wmo_w_indexing.db")
sites = [
    (r.met_station.lon, r.met_station.lat, r.temperature)
    for r in db.query(DbWMOMetDailyTempMean)
    .filter(DbWMOMetDailyTempMean.date_int == date_to_int(date))
    .all()
]
points = np.array([r[:-1] for r in sites])
px = points[:, 0]
py = points[:, 1]
pxm, pym = eg.ease1_lonlat_to_meters(points[:, 0], points[:, 1])
pm = np.array(list(zip(pxm, pym)))
values = np.array([int(s[-1] > 273.15) for s in sites])
lons, lats = eg.ease1_get_full_grid_lonlat(eg.ML)
xm, ym = eg.ease1_lonlat_to_meters(lons, lats, eg.ML)
ip = NearestNDInterpolator(pm, values)
igrid = ip(xm, ym)
dist, _ = ip.tree.query(np.array(list(zip(xm.ravel(), ym.ravel()))))
dist = dist.reshape(xm.shape)

cmap = cmap = colors.ListedColormap(["skyblue", "lightcoral"])
norm = colors.BoundaryNorm([0, 1, 2], 2)

# Plot interpolation
ax = plt.axes(projection=ccrs.EckertVI())
plt.contourf(
    lons, lats, igrid, 2, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm
)
ax.coastlines()
plt.title("WMO 2000-01-01: Neareset Neighbor Interpolation")

plt.figure()
ax = plt.axes(projection=ccrs.EckertVI())
plt.contourf(lons, lats, dist, 50, transform=ccrs.PlateCarree())
plt.title("WMO 2000-01-01: Neareset Neighbor Distance Field")
plt.show()
