"""
Extract WorldPop population (exposure layer) to the Chennai 100 m grid

"""

import os, numpy as np, pandas as pd, rasterio
from rasterio.warp import transform

PROJ = r"C:\毕设项目"
HEAT   = os.path.join(PROJ, r"data\heat\chennai_heatsusc_100m.tif")     # master grid
POP_IN = os.path.join(PROJ, r"data\composite\ind_pop_2020_CN_100m_R2025A_v1.tif")
OUT    = os.path.join(PROJ, r"data\composite")


# 1. Master-grid valid-cell centroids (EPSG:32644)
with rasterio.open(HEAT) as s:
    heat = s.read(1).astype("float32")
    T, W, H, NOD, CRS, PROF = s.transform, s.width, s.height, s.nodata, s.crs, s.profile
valid = np.isfinite(heat) & (heat != NOD)
rows, cols = np.where(valid)
xs, ys = rasterio.transform.xy(T, rows, cols)
xs, ys = np.array(xs), np.array(ys)

# 2. Project centroids to lon/lat and point-sample WorldPop
lon, lat = transform("EPSG:32644", "EPSG:4326", xs, ys)
lon, lat = np.array(lon), np.array(lat)
print(f"AOI  lon {lon.min():.3f}-{lon.max():.3f}   lat {lat.min():.3f}-{lat.max():.3f}")

with rasterio.open(POP_IN) as s:
    nd = s.nodata
    pop = np.array([v[0] for v in s.sample(zip(lon, lat))], dtype="float64")
pop[pop == nd] = np.nan
pop[pop < 0]   = np.nan

m = np.isfinite(pop)
print(f"cells {len(pop)} | valid {m.sum()} | total pop {np.nansum(pop):,.0f} | "
      f"per-cell mean {np.nanmean(pop):.1f} median {np.nanmedian(pop):.1f} "
      f"max {np.nanmax(pop):.1f} CV {np.nanstd(pop)/np.nanmean(pop):.2f}")


# 3. Write aligned population GeoTIFF
grid = np.full((H, W), -9999.0, dtype="float32")
grid[rows[m], cols[m]] = pop[m]
PROF.update(dtype="float32", nodata=-9999.0, count=1)
with rasterio.open(os.path.join(OUT, "chennai_pop_2020_100m.tif"), "w", **PROF) as d:
    d.write(grid, 1)

pd.DataFrame({"x": xs, "y": ys, "lon": lon, "lat": lat, "pop": pop}) \
  .to_csv(os.path.join(OUT, "chennai_pop_2020_cells.csv"), index=False)
print("wrote chennai_pop_2020_100m.tif + chennai_pop_2020_cells.csv")
