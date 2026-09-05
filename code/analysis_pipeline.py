import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Read table
CSV = r"C:\毕设项目\data\heat\chennai_heat_variables_table.csv"
df = pd.read_csv(CSV)
 
cols = ['LST','NDVI','albedo','builtup','dist_water','NDBI','bare','bheight']
d = df[cols].copy()
 
print("rows:", len(d))
print("\nmissing values:\n", d.isna().sum())
print("\ndescriptive statistics:\n", d.describe().T[['mean','std','min','max']].round(3))
 
# 2. Spearman correlation matrix
sp = d.corr(method='spearman')
print("\n=== Spearman correlation matrix ===\n", sp.round(2))
 
# 3. Direction check: each variable vs LST
print("\n=== vs LST (direction) ===")
print(sp['LST'].drop('LST').sort_values(ascending=False).round(2))
 
# Optional: partial correlation of albedo with LST controlling for built-up
import pingouin as pg
print(pg.partial_corr(data=d, x='albedo', y='LST', covar='builtup', method='spearman'))
 
# 4. Find redundant pairs (|rho| > 0.8, excluding self and LST)
print("\n=== redundant pairs |rho|>0.8 (among modifier variables) ===")
mods = [c for c in cols if c != 'LST']
sub = sp.loc[mods, mods]
seen = set()
for i in mods:
    for j in mods:
        if i < j and abs(sub.loc[i,j]) > 0.8:
            print(f"  {i} <-> {j}: {sub.loc[i,j]:.2f}")
 
# 5. Heatmap
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(sp, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{sp.iloc[i,j]:.2f}", ha='center', va='center',
                color='white' if abs(sp.iloc[i,j])>0.5 else 'black', fontsize=8)
fig.colorbar(im, label='Spearman rho')
plt.title('Heat candidates - Spearman correlation')
plt.tight_layout()
plt.savefig(r"C:\毕设项目\data\heat\spearman_matrix.png", dpi=150)
plt.show()
print("\nheatmap saved: data/heat/spearman_matrix.png")
 
 
 
 
import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Point
 
# 1. Read heat variable table, parse lon/lat from .geo
df = pd.read_csv(r"C:\毕设项目\data\heat\chennai_heat_variables_table.csv")
coords = df['.geo'].apply(lambda s: json.loads(s)['coordinates'])
df['lon'] = coords.apply(lambda c: c[0])
df['lat'] = coords.apply(lambda c: c[1])
 
gdf = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df['lon'], df['lat'])],
    crs='EPSG:4326')
 
# 2. Read slum boundaries (887 polygons, also EPSG:4326)
slum = gpd.read_file(r"C:\毕设项目\data\slum_boundary_shapefile\output.shp")[['geometry']]
 
# 3. Spatial join: point inside any slum polygon = 1
joined = gpd.sjoin(gdf, slum, how='left', predicate='within')
df['slum'] = joined.index_right.notna().astype(int).groupby(level=0).max().values
 
# 4. Check
print("total cells:", len(df))
print("slum:", int(df['slum'].sum()), f"({df['slum'].mean()*100:.1f}%)")
print("non-slum:", int((df['slum']==0).sum()))
print("\nslum vs non-slum mean LST:")
print(df.groupby('slum')['LST'].agg(['mean','std','count']).round(3))
 
# 5. Save new table with slum labels
out = r"C:\毕设项目\data\heat\chennai_heat_variables_tagged.csv"
df.drop(columns='.geo').to_csv(out, index=False)
print("\nsaved:", out)
 
 

# Scheme B: heat susceptibility index
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from scipy.optimize import minimize
np.random.seed(0)
 
PATH = r"C:\毕设项目\data\heat\chennai_heat_variables_tagged.csv"
df = pd.read_csv(PATH)
directed = [('NDVI','-'),('builtup','+'),('dist_water','+')]
TARGET = 'LST'
 
def norm_mm(col, sign):
    x=df[col]; lo,hi=x.quantile(.01),x.quantile(.99)
    z=((x-lo)/(hi-lo)).clip(0,1); return z if sign=='+' else 1-z
def norm_rank(col, sign):
    r=df[col].rank(pct=True); return r if sign=='+' else 1-r
def build(weights, normfn):
    w=np.array(weights,float); w=w/w.sum()
    return sum(normfn(c,s)*wi for (c,s),wi in zip(directed,w))
def eval_slum(idx):
    s=idx[df.slum==1]; ns=idx[df.slum==0]
    U,p=mannwhitneyu(s,ns,alternative='greater'); auc=U/(len(s)*len(ns))
    return auc, 2*auc-1, s.mean()-ns.mean(), p
 
idx0 = build([1,1,1], norm_mm)
 
rho_idx,p_idx = spearmanr(idx0, df[TARGET])
print("=== Validation 1: driver index vs observed LST (Spearman) ===")
print(f"  rho = {rho_idx:.3f}  (p={p_idx:.1e})  degree to which the index reproduces the real heat pattern")
print("  single variable vs LST reference:")
for c,s in directed:
    r,_=spearmanr(norm_mm(c,s), df[TARGET]); print(f"    {c:<11} rho = {r:.3f}")
 
print("\n=== Validation 2: ablation (leave-one-out, target = reproduce LST) ===")
full=abs(rho_idx)
for i,(c,s) in enumerate(directed):
    sub=[d for j,d in enumerate(directed) if j!=i]
    idx_sub=sum(norm_mm(cc,ss) for cc,ss in sub)/len(sub)
    r,_=spearmanr(idx_sub, df[TARGET]); print(f"  remove {c:<11} rho={abs(r):.3f}  delta={abs(r)-full:+.3f}")
 
# Validation 3: equal weight vs fitted optimum (grid search of weights maximising Spearman vs LST)
best=-9.0; bw=None
for a in range(0,21):
    for b in range(0,21-a):
        c=20-a-b
        w=(a/20.,b/20.,c/20.)
        r=abs(spearmanr(build([w[0] or 1e-9,w[1] or 1e-9,w[2] or 1e-9],norm_mm),df[TARGET])[0])
        if r>best: best=r; bw=w
print("\n=== Validation 3: equal weight vs fitted optimum (grid search) ===")
print(f"  equal weight   rho = {full:.3f}")
print(f"  fitted optimum rho = {best:.3f}   weights NDVI/builtup/dist_water = {bw[0]:.2f}/{bw[1]:.2f}/{bw[2]:.2f}")
print(f"  equal weight minus optimum = {full-best:+.3f}")
 
auc,cd,gap,p=eval_slum(idx0)
print("\n=== Main result: slum vs non-slum (equal-weight driver index) ===")
print(f"  slum mean={idx0[df.slum==1].mean():.3f}  non-slum mean={idx0[df.slum==0].mean():.3f}")
print(f"  AUC={auc:.3f}  Cliff d={cd:.3f}  gap={gap:+.3f}  p={p:.1e}")
 
print("\n=== Sensitivity 1: weight schemes ===")
schemes={'equal':[1,1,1],'builtup-heavy':[.2,.6,.2],'NDVI-heavy':[.6,.2,.2],'no dist_water':[1,1,0],'no NDVI':[0,1,1]}
for name,w in schemes.items():
    a,c,g,_=eval_slum(build(w,norm_mm)); print(f"  {name:<15} AUC={a:.3f} gap={g:+.3f}")
 
aucs=np.array([eval_slum(build(np.random.dirichlet(np.ones(3)),norm_mm))[0] for _ in range(500)])
print("\n=== Sensitivity 2: 500 random weight sets (Dirichlet) ===")
print(f"  AUC min={aucs.min():.3f} median={np.median(aucs):.3f} max={aucs.max():.3f}")
print(f"  slum hotter share: {(aucs>0.5).mean()*100:.1f}%")
 
print("\n=== Sensitivity 3: normalisation ===")
for nm,fn in [('min-max',norm_mm),('rank',norm_rank)]:
    a,c,g,_=eval_slum(build([1,1,1],fn)); print(f"  {nm:<10} AUC={a:.3f} gap={g:+.3f}")
 
df['heat_susc_index']=idx0
out=r"C:\毕设项目\data\heat\chennai_heat_index_driverB.csv"
df.to_csv(out,index=False); print(f"\nsaved heat_susc_index column -> chennai_heat_index_driverB.csv")
 
 
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
 
DIR  = r"C:\毕设项目\data\heat"
CSV  = DIR + r"\chennai_heat_index_driverB.csv"
df   = pd.read_csv(CSV).dropna(subset=['lon', 'lat', 'heat_susc_index', 'LST'])
lon, lat = df['lon'].values, df['lat'].values
ASP  = 1.0 / np.cos(np.deg2rad(float(lat.mean())))
 
# 1. PNG (scatter raster, quick visualisation)
def draw(vals, cmap, title, label, fname, vmin, vmax, overlay=False):
    fig, ax = plt.subplots(figsize=(7, 8.5))
    sc = ax.scatter(lon, lat, c=np.asarray(vals, float), cmap=cmap,
                    vmin=vmin, vmax=vmax, s=9, marker='s', linewidths=0)
    if overlay:
        sl = df[df.slum == 1]
        ax.scatter(sl['lon'], sl['lat'], s=9, facecolors='none',
                   edgecolors='black', linewidths=0.4, label='slum cells')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=11); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_aspect(ASP)
    fig.colorbar(sc, ax=ax, shrink=0.65, label=label)
    plt.tight_layout(); plt.savefig(DIR + '\\' + fname, dpi=150); plt.close()
    print('saved', fname)
 
p2, p98 = np.nanpercentile(df['LST'], 2), np.nanpercentile(df['LST'], 98)
draw(df['heat_susc_index'], 'RdYlBu_r',
     'Chennai heat susceptibility index\n(equal-weight drivers: NDVI-, built-up+, dist_water+)',
     'heat susceptibility (0 cool - 1 hot)', 'map_heat_susc_index.png', 0, 1)
draw(df['LST'], 'inferno', 'Chennai hot-season LST (observed)', 'LST (deg C)',
     'map_LST_observed.png', p2, p98)
draw(df['heat_susc_index'], 'RdYlBu_r', 'Heat susceptibility with slum cells overlaid',
     'heat susceptibility', 'map_heat_susc_slum_overlay.png', 0, 1, overlay=True)
 
# 2. GeoTIFF (EPSG:32644, 100m, overlayable in GIS)
tr = Transformer.from_crs('EPSG:4326', 'EPSG:32644', always_xy=True)
x, y = tr.transform(lon, lat); x, y = np.asarray(x), np.asarray(y)
RES = 100.0; x0, y1 = x.min(), y.max()
col = np.round((x - x0) / RES).astype(int)
row = np.round((y1 - y) / RES).astype(int)
W, H = col.max() + 1, row.max() + 1
transform = from_origin(x0 - RES/2, y1 + RES/2, RES, RES)   # top-left corner = cell centre - half cell
NODATA = -9999.0
 
def grid(v):
    g = np.full((H, W), NODATA, dtype='float32')
    g[row, col] = np.where(np.isfinite(v), v, NODATA).astype('float32')
    return g
 
def write_tif(fname, band_dict):
    names = list(band_dict)
    with rasterio.open(DIR + '\\' + fname, 'w', driver='GTiff', height=H, width=W,
                       count=len(names), dtype='float32', crs='EPSG:32644',
                       transform=transform, nodata=NODATA, compress='deflate') as ds:
        for i, n in enumerate(names, 1):
            ds.write(grid(band_dict[n].values.astype(float)), i)
            ds.set_band_description(i, n)
    print('saved', fname)
 
write_tif('chennai_heatsusc_100m.tif',            {'heat_susc_index': df['heat_susc_index']})
write_tif('chennai_LST_hotseason_100m_FROMCSV.tif', {'LST_hotseason_degC': df['LST']})
write_tif('chennai_heatstack_100m.tif', {
    'heat_susc_index': df['heat_susc_index'], 'LST': df['LST'], 'NDVI': df['NDVI'],
    'builtup': df['builtup'], 'dist_water': df['dist_water'], 'slum': df['slum']})
 
print('\ndone: 3 PNG + 3 GeoTIFF, all in', DIR)
 
 


# Flood candidate variable screening
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
 
DIR = r"C:\毕设项目\data\flood"
df  = pd.read_csv(DIR + r"\chennai_flood_variables_table.csv")
 
# .geo -> lon/lat
coords = df['.geo'].apply(lambda s: json.loads(s)['coordinates'])
df['lon'] = coords.apply(lambda c: c[0]); df['lat'] = coords.apply(lambda c: c[1])
# flow_accum strongly right-skewed -> log (only for later min-max normalisation; note Spearman is rank-based, log does not change ranks)
df['log_flow_accum'] = np.log1p(df['flow_accum'])
 
cols = ['elevation','slope','HAND','flow_accum','TWI','dist_water','builtup','impervious']
print('n =', len(df))
print('\n=== descriptive statistics ===')
print(df[cols].describe().T[['mean','std','min','50%','max']].round(3))
 
sp = df[cols].corr(method='spearman')
print('\n=== Spearman correlation matrix ===')
print(sp.round(2).to_string())
 
print('\n=== redundant pairs |rho|>0.8 ===')
for i in cols:
    for j in cols:
        if i < j and abs(sp.loc[i, j]) > 0.8:
            print(f'  {i} <-> {j}: {sp.loc[i,j]:.2f}')
 
# theoretical direction 
direction = {'elevation':'-','slope':'-','HAND':'-','flow_accum':'+','TWI':'+',
             'dist_water':'-','builtup':'+','impervious':'+'}
print('\n=== theoretical direction (pending vs observed inundation validation) ==='); print(direction)
 
# Heatmap
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sp, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{sp.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(sp.iloc[i,j]) > 0.5 else 'black')
fig.colorbar(im, label='Spearman rho'); plt.title('Flood candidates - Spearman correlation')
plt.tight_layout(); plt.savefig(DIR + r"\flood_spearman_matrix.png", dpi=150)
df.drop(columns='.geo').to_csv(DIR + r"\chennai_flood_variables_clean.csv", index=False)
print('\nsaved chennai_flood_variables_clean.csv + flood_spearman_matrix.png')
 
 

# Flood layer
import pandas as pd, numpy as np, matplotlib, rasterio
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from pyproj import Transformer
from scipy.stats import mannwhitneyu
 
FLOOD = r"C:\毕设项目\data\flood"
HEAT  = r"C:\毕设项目\data\heat"
fl = pd.read_csv(FLOOD + r"\chennai_flood_variables_tagged.csv")     # contains flood_haz(1-5)
ht = pd.read_csv(HEAT  + r"\chennai_heat_variables_tagged.csv")[['lon','lat','slum']]
for d in (fl, ht): d['k'] = d['lon'].round(5).astype(str)+'_'+d['lat'].round(5).astype(str)
df = fl.merge(ht[['k','slum']], on='k', how='left').dropna(subset=['lon','lat'])
df['flood_idx'] = (df['flood_haz'] - 1) / 4      # normalise 0-1
lon, lat = df['lon'].values, df['lat'].values
ASP = 1/np.cos(np.deg2rad(lat.mean()))
 
# slum vs non-slum (GCC zoning)
sub = df[df.flood_idx.notna() & df.slum.notna()]
s, ns = sub[sub.slum==1]['flood_idx'], sub[sub.slum==0]['flood_idx']
U,p = mannwhitneyu(s, ns, alternative='greater'); auc = U/(len(s)*len(ns))
print('=== flood hazard (GCC) slum vs non-slum ===')
print('slum mean=%.3f | non-slum mean=%.3f | gap=%+.3f | AUC=%.3f | p=%.1e'
      % (s.mean(), ns.mean(), s.mean()-ns.mean(), auc, p))
 
# Mapping
def draw(cmap, title, fname, overlay=False):
    fig, ax = plt.subplots(figsize=(6.8, 8.3))
    sc = ax.scatter(sub.lon, sub.lat, c=sub.flood_idx, cmap=cmap, vmin=0, vmax=1, s=9, marker='s', linewidths=0)
    ax.scatter(df[df.flood_idx.isna()].lon, df[df.flood_idx.isna()].lat, c='lightgray', s=6, marker='s', linewidths=0)
    if overlay:
        sl = df[df.slum==1]; ax.scatter(sl.lon, sl.lat, s=9, facecolors='none', edgecolors='red', linewidths=0.4, label='slum cells')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect(ASP); ax.set_title(title); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    fig.colorbar(sc, ax=ax, shrink=0.65, label='flood hazard (0-1)')
    plt.tight_layout(); plt.savefig(FLOOD + '\\' + fname, dpi=150); plt.close(); print('saved', fname)
 
draw('Blues', 'Chennai flood hazard (GCC official inundation zones, 0-1)', 'map_flood_hazard.png')
draw('Blues', 'Flood hazard (GCC zones) with slum cells overlaid', 'map_flood_hazard_slum_overlay.png', overlay=True)
 
# GeoTIFF (EPSG:32644)
tr = Transformer.from_crs('EPSG:4326', 'EPSG:32644', always_xy=True)
x, y = tr.transform(lon, lat); x, y = np.asarray(x), np.asarray(y)
RES = 100.0; x0, y1 = x.min(), y.max()
col = np.round((x-x0)/RES).astype(int); row = np.round((y1-y)/RES).astype(int)
W, H = col.max()+1, row.max()+1; NOD = -9999.0
g = np.full((H, W), NOD, 'float32'); g[row, col] = np.where(np.isfinite(df.flood_idx), df.flood_idx, NOD).astype('float32')
with rasterio.open(FLOOD + r"\chennai_floodhazard_100m.tif", 'w', driver='GTiff', height=H, width=W, count=1,
                   dtype='float32', crs='EPSG:32644', transform=from_origin(x0-RES/2, y1+RES/2, RES, RES),
                   nodata=NOD, compress='deflate') as ds:
    ds.write(g, 1); ds.set_band_description(1, 'flood_hazard_GCC_0_1')
print('saved chennai_floodhazard_100m.tif')
 
 
#OSM processing

import glob, re, numpy as np, pandas as pd, geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import get_coordinates
 
DIR   = r"C:\毕设项目\data\pollution"
GRID  = r"C:\毕设项目\data\heat\chennai_heat_variables_tagged.csv"   # same 100m grid (lon/lat/slum)
GPKG  = glob.glob(DIR + r"\**\*.gpkg", recursive=True)[0]
print("gpkg:", GPKG)
 
# grid centres 
g = pd.read_csv(GRID)[['lon','lat','slum']].dropna().reset_index(drop=True)
tr = Transformer.from_crs('EPSG:4326','EPSG:32644', always_xy=True)
gx, gy = tr.transform(g.lon.values, g.lat.values)
 
# 1. Road density (weighted by highway type, 100m histogram)
w = {'motorway':5,'trunk':5,'motorway_link':4,'trunk_link':4,'primary':4,'primary_link':3,
     'secondary':3,'secondary_link':2,'tertiary':2,'tertiary_link':2,'residential':1,
     'living_street':1,'unclassified':1,'service':0.5}
ln = gpd.read_file(GPKG, layer='lines', columns=['highway','geometry'])
ln = ln[ln['highway'].isin(w)].to_crs('EPSG:32644')
ln['wt'] = ln['highway'].map(w)
# sample a point every 50m along the line 
px, py, pw = [], [], []
for geom, wt in zip(ln.geometry.values, ln['wt'].values):
    if geom is None: continue
    n = max(int(geom.length // 50), 1)
    for dd in np.linspace(0, geom.length, n+1):
        p = geom.interpolate(dd); px.append(p.x); py.append(p.y); pw.append(wt)
px, py, pw = np.array(px), np.array(py), np.array(pw)
RES = 100.0
x0 = min(gx.min(), px.min()) - RES; y0 = min(gy.min(), py.min()) - RES
xe = np.arange(x0, max(gx.max(), px.max()) + 2*RES, RES)
ye = np.arange(y0, max(gy.max(), py.max()) + 2*RES, RES)
H, _, _ = np.histogram2d(px, py, bins=[xe, ye], weights=pw)
ci = np.clip(np.searchsorted(xe, gx) - 1, 0, H.shape[0]-1)
ri = np.clip(np.searchsorted(ye, gy) - 1, 0, H.shape[1]-1)
g['road_density'] = H[ci, ri]
 
# 2. Distance to industry
mp = gpd.read_file(GPKG, layer='multipolygons', columns=['name','other_tags','geometry'])
ot = mp['other_tags'].astype(str)
ind = mp[ot.str.contains('industrial', case=False, na=False) |
         mp['name'].astype(str).str.contains('Industrial|Refinery|Estate|SIDCO|SIPCOT', case=False, na=False)].to_crs('EPSG:32644')
ic = get_coordinates(ind.geometry.values)
d, _ = cKDTree(ic).query(np.c_[gx, gy]); g['dist_industry'] = d
print('industrial features:', len(ind))
 
g.to_csv(DIR + r"\osm_road_industry.csv", index=False)
print('road_density: min %.1f max %.1f CV %.0f%% | dist_industry(m): %.0f-%.0f'
      % (g.road_density.min(), g.road_density.max(), g.road_density.std()/g.road_density.mean()*100,
         g.dist_industry.min(), g.dist_industry.max()))
print('saved osm_road_industry.csv, n=', len(g))
 
 
#Pollution candidate variable screening

import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
import xarray as xr
 
DIR    = r"C:\毕设项目\data\pollution"
XLSX   = DIR + r"\chennai_pollution_variables_table.xlsx"
OSM    = DIR + r"\osm_road_industry.csv"
NO2_NC = r"C:\毕设项目\data\pollution\TROPOMI-inferred_surface_no2_asia_2019_annual_mean.nc"
 
# 1. Merge candidate pool (GEE + OSM) onto a common grid
xl = pd.read_excel(XLSX)
co = xl['.geo'].apply(lambda s: json.loads(s)['coordinates'])
xl['lon'] = co.apply(lambda c: round(c[0],5)); xl['lat'] = co.apply(lambda c: round(c[1],5))
osm = pd.read_csv(OSM); osm['lon'] = osm.lon.round(5); osm['lat'] = osm.lat.round(5)
df = xl.merge(osm[['lon','lat','slum','road_density','dist_industry']], on=['lon','lat'], how='inner')
print('cells after merge:', len(df))
 
# 2. Sample validation target surface NO2 (LUR ground concentration, not in index)
ds = xr.open_dataset(NO2_NC, engine='h5netcdf')
la = ds['LAT_CENTER'].values.ravel(); lo = ds['LON_CENTER'].values.ravel()
iy = np.where((la>=12.85)&(la<=13.25))[0]; ix = np.where((lo>=80.05)&(lo<=80.40))[0]
win = ds['surface_no2_ppb'].values[iy.min():iy.max()+1, ix.min():ix.max()+1].astype(float)
wla = la[iy.min():iy.max()+1]; wlo = lo[ix.min():ix.max()+1]
df['NO2_surface'] = [win[np.abs(wla-a).argmin(), np.abs(wlo-o).argmin()] for o,a in zip(df.lon, df.lat)]
df = df[np.isfinite(df.NO2_surface)]
 
# driver candidates (relative "more polluted"): road_density+ built-up+ VIIRS+ NDBI+ NDVI- dist_industry-
directed = [('road_density','+'),('builtup','+'),('VIIRS','+'),('NDBI','+'),('NDVI','-'),('dist_industry','-')]
cols = [c for c,_ in directed]
 
print('\n=== descriptive statistics ===')
print(df[cols].describe().T[['mean','std','min','50%','max']].round(3).to_string())
 
# 3. Redundancy: Spearman matrix + heatmap (first, same as heat/flood)
sp = df[cols].corr('spearman')
print('\n=== Spearman redundancy matrix ===\n', sp.round(2).to_string())
print('\nredundant pairs |rho|>0.8:')
for i in cols:
    for j in cols:
        if i<j and abs(sp.loc[i,j])>0.8: print('  %s <-> %s: %.2f'%(i,j,sp.loc[i,j]))
fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(sp,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=45,ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j,i,'%.2f'%sp.iloc[i,j],ha='center',va='center',fontsize=8,
                color='white' if abs(sp.iloc[i,j])>0.5 else 'black')
fig.colorbar(im,label='Spearman rho'); plt.title('Pollution candidates - Spearman')
plt.tight_layout(); plt.savefig(DIR + r"\pollution_spearman_matrix.png", dpi=150)
 
# 4. Direction check: each driver vs surface NO2
print('\n=== direction check: drivers vs surface NO2 (N=%d) ==='%len(df))
for c,s in directed:
    rho,_=spearmanr(df[c],df.NO2_surface); ok='OK' if (rho>0)==(s=='+') else 'XX rev'
    print('  %-13s rho=%+.3f  expected%s  %s'%(c,rho,s,ok))
 
# 5. (after screening) equal-weight driver index + validation + slum
final=[('road_density','+'),('builtup','+'),('VIIRS','+'),('NDBI','+')]   # drop NDVI (redundant + wrong sign) / dist_industry (wrong sign + sparse)
def nm(c,s):
    x=df[c]; lo2,hi2=x.quantile(.01),x.quantile(.99); z=((x-lo2)/(hi2-lo2)).clip(0,1); return z if s=='+' else 1-z
df['poll_idx']=sum(nm(c,s) for c,s in final)/len(final)
r,_=spearmanr(df.poll_idx, df.NO2_surface)
print('\n=== equal-weight driver index (road/built-up/VIIRS/NDBI) vs surface NO2: rho=%.3f ==='%r)
sl=df[df.slum==1].poll_idx; ns=df[df.slum==0].poll_idx
U,p=mannwhitneyu(sl,ns,alternative='greater'); auc=U/(len(sl)*len(ns))
print('=== slum vs non-slum: slum %.3f non-slum %.3f gap %+.3f AUC %.3f p %.1e ==='
      %(sl.mean(),ns.mean(),sl.mean()-ns.mean(),auc,p))
df.drop(columns='.geo',errors='ignore').to_csv(DIR + r"\chennai_pollution_screened.csv", index=False)
print('\nsaved pollution_spearman_matrix.png + chennai_pollution_screened.csv')

# 6. Write pollution index to the same 100 m raster grid used by the heat layer.
import rasterio
from pyproj import Transformer

HEAT_REF = r"C:\毕设项目\data\heat\chennai_heatsusc_100m.tif"
with rasterio.open(HEAT_REF) as src:
    prof = src.profile.copy()
    transform = src.transform
    width, height = src.width, src.height
    nodata = src.nodata if src.nodata is not None else -9999.0

prof.update(count=1, dtype="float32", nodata=nodata)
poll_grid = np.full((height, width), nodata, dtype="float32")
tr = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
x, y = tr.transform(df["lon"].values, df["lat"].values)
cc, rr = (~transform) * (x, y)
cc = np.floor(cc).astype(int)
rr = np.floor(rr).astype(int)
ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width) & np.isfinite(df["poll_idx"].values)
poll_grid[rr[ok], cc[ok]] = df["poll_idx"].values[ok].astype("float32")

poll_tif = DIR + r"\chennai_pollution_index_100m.tif"
with rasterio.open(poll_tif, "w", **prof) as dst:
    dst.write(poll_grid, 1)
    dst.set_band_description(1, "pollution_index")

plot_df = df.loc[ok].copy()
plt.figure(figsize=(6, 7))
plt.scatter(plot_df["lon"], plot_df["lat"], c=plot_df["poll_idx"], s=2, cmap="magma", linewidths=0)
plt.gca().set_aspect("equal", adjustable="box")
plt.colorbar(label="pollution index")
plt.title("Chennai pollution susceptibility index")
plt.tight_layout()
plt.savefig(DIR + r"\map_pollution_index.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 7))
plt.scatter(plot_df["lon"], plot_df["lat"], c=plot_df["poll_idx"], s=2, cmap="magma", linewidths=0)
slum_df = plot_df[plot_df["slum"] == 1]
plt.scatter(slum_df["lon"], slum_df["lat"], s=4, facecolors="none", edgecolors="cyan", linewidths=0.4)
plt.gca().set_aspect("equal", adjustable="box")
plt.colorbar(label="pollution index")
plt.title("Pollution susceptibility with slum cells overlaid")
plt.tight_layout()
plt.savefig(DIR + r"\map_pollution_slum_overlay.png", dpi=150)
plt.close()
print("saved chennai_pollution_index_100m.tif + map_pollution_index.png + map_pollution_slum_overlay.png")
 

 


# Composite multi-hazard index (MHI) - Level 2

 


import os, numpy as np, pandas as pd, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import rankdata, mannwhitneyu
 
PROJ = r"C:\毕设项目"
HEAT  = os.path.join(PROJ, r"data\heat\chennai_heatsusc_100m.tif")
FLOOD = os.path.join(PROJ, r"data\flood\chennai_floodhazard_100m.tif")
POLL  = os.path.join(PROJ, r"data\pollution\chennai_pollution_index_100m.tif")
SLUM  = os.path.join(PROJ, r"data\slum_boundary_shapefile\output.shp")
OUT   = os.path.join(PROJ, r"data\composite")
os.makedirs(OUT, exist_ok=True)
 
SEED = 42
np.random.seed(SEED)
 
 

# 1. Load the heat raster as the MASTER grid; get valid-cell centroids
with rasterio.open(HEAT) as s:
    heat_arr = s.read(1).astype("float32")
    T, W, H  = s.transform, s.width, s.height
    NOD, CRS = s.nodata, s.crs
    PROF     = s.profile
 
valid = np.isfinite(heat_arr) & (heat_arr != NOD)
rows, cols = np.where(valid)
xs, ys = rasterio.transform.xy(T, rows, cols)
xs, ys = np.array(xs), np.array(ys)
heat = heat_arr[rows, cols]                       # 1-D, master cell order
print(f"master (heat-valid) cells: {len(xs)}")
 
 

# 2. Sample flood & pollution rasters at the master centroids
def sample_raster(path, xs, ys):
    with rasterio.open(path) as s:
        arr, tt, nd, w, h = s.read(1).astype("float32"), s.transform, s.nodata, s.width, s.height
    inv = ~tt
    c = inv * (xs, ys)
    cc, rr = np.floor(c[0]).astype(int), np.floor(c[1]).astype(int)
    out = np.full(xs.shape, np.nan, dtype="float32")
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    out[ok] = arr[rr[ok], cc[ok]]
    out[out == nd] = np.nan
    return out
 
flood = sample_raster(FLOOD, xs, ys)
poll  = sample_raster(POLL,  xs, ys)
print(f"flood present: {np.isfinite(flood).sum()}  | pollution present: {np.isfinite(poll).sum()}")
 
 

# 3. Tag slum cells (cell centroid within any slum polygon)
slum_gdf = gpd.read_file(SLUM).to_crs(CRS)
pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=CRS)
j = gpd.sjoin(pts, slum_gdf[["geometry"]], how="left", predicate="within")
slum = (~j.index_right.isna()).groupby(j.index).max() \
        .reindex(range(len(xs))).fillna(False).values.astype(bool)
print(f"slum cells: {slum.sum()} ({100*slum.mean():.1f}%)")
 
 

# 4. Helper functions
def minmax(a, mask):
    lo, hi = np.nanmin(a[mask]), np.nanmax(a[mask])
    return (a - lo) / (hi - lo)
 
def auc(score, lab):
    """Probability a random slum cell outranks a random non-slum cell."""
    r = rankdata(score); n1, n0 = lab.sum(), (~lab).sum()
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
 
def hotspot_ratio(score, lab, pct):
    thr = np.percentile(score, 100 - pct)
    hot = score >= thr
    s, n = hot[lab].mean(), hot[~lab].mean()
    return s, n, (s / n if n > 0 else np.nan)
 
 

# 5. Build the equal-weight MHI on complete-coverage cells
complete = np.isfinite(heat) & np.isfinite(flood) & np.isfinite(poll)
print(f"complete (3-layer) cells: {complete.sum()}  | slum among them: {slum[complete].sum()}")
 
hN = minmax(heat,  complete)     # re-normalise each layer 0-1 on the common cells
fN = minmax(flood, complete)
pN = minmax(poll,  complete)
 
MHI   = (hN + fN + pN) / 3.0     # equal weights (baseline)
MHI_n = minmax(MHI, complete)    # 0-1 for display
 
 

# 6. Slum vs non-slum: per-layer and composite (Table 4.4)
def compare(name, score):
    sm, ns = score[complete & slum], score[complete & ~slum]
    U, p = mannwhitneyu(sm, ns, alternative="two-sided")
    a = auc(score[complete], slum[complete])
    print(f"{name:12s} slum={sm.mean():.3f} nonslum={ns.mean():.3f} "
          f"med {np.median(sm):.3f}/{np.median(ns):.3f}  AUC={a:.3f}  p={p:.2e}")
    return a
 
print("\n=== Table 4.4  slum vs non-slum ===")
compare("Heat",      hN)
compare("Flood(GCC)", fN)
compare("Pollution", pN)
compare("MHI(equal)", MHI_n)
 
 

# 7. Hotspot over-exposure (equal-weight MHI, and heat+pollution for contrast)
print("\n=== Hotspot over-exposure (slum / non-slum share) ===")
print("equal-weight MHI:")
for pct in (10, 20, 30):
    s, n, r = hotspot_ratio(MHI_n[complete], slum[complete], pct)
    print(f"  top-{pct:>2}%: slum {100*s:.1f}%  non-slum {100*n:.1f}%  = {r:.2f}x")
 
HP = minmax((hN + pN) / 2, complete)   # flood removed -> the diluting layer omitted
print("heat+pollution only (flood removed):")
for pct in (10, 20, 30):
    s, n, r = hotspot_ratio(HP[complete], slum[complete], pct)
    print(f"  top-{pct:>2}%: slum {100*s:.1f}%  non-slum {100*n:.1f}%  = {r:.2f}x")
 
 

# 8. Sensitivity analysis
print("\n=== A. weight scenarios (slum AUC on MHI) ===")
for name, (wh, wf, wp) in {
        "equal        (1/3,1/3,1/3)": (1/3, 1/3, 1/3),
        "heat-heavy    (.5,.25,.25)": (.5, .25, .25),
        "flood-heavy   (.25,.5,.25)": (.25, .5, .25),
        "pollutn-heavy (.25,.25,.5)": (.25, .25, .5)}.items():
    m = minmax(wh*hN + wf*fN + wp*pN, complete)
    print(f"  {name}: AUC={auc(m[complete], slum[complete]):.3f}")
 
print("\n=== B. 500 Dirichlet random weights ===")
aucs = []
for _ in range(500):
    w = np.random.dirichlet([1, 1, 1])
    m = minmax(w[0]*hN + w[1]*fN + w[2]*pN, complete)
    aucs.append(auc(m[complete], slum[complete]))
aucs = np.array(aucs)
print(f"  AUC mean={aucs.mean():.3f} sd={aucs.std():.3f} "
      f"range[{aucs.min():.3f},{aucs.max():.3f}]  %>0.5={100*(aucs>0.5).mean():.0f}%")
 
print("\n=== C. rank-normalisation variant ===")
hr = minmax(rankdata(hN[complete]).astype(float), np.ones(complete.sum(), bool))
fr = minmax(rankdata(fN[complete]).astype(float), np.ones(complete.sum(), bool))
pr = minmax(rankdata(pN[complete]).astype(float), np.ones(complete.sum(), bool))
mr = (hr + fr + pr) / 3
print(f"  rank-norm equal-weight MHI slum AUC={auc(mr, slum[complete]):.3f}")
 
print("\n=== D. robustness: heat+pollution composite on FULL grid ===")
hp_mask = np.isfinite(heat) & np.isfinite(poll)      # keeps unclassified-flood cells
h2 = minmax(heat, hp_mask); p2 = minmax(poll, hp_mask)
HPfull = minmax((h2 + p2) / 2, hp_mask)
print(f"  cells={hp_mask.sum()}  slum AUC={auc(HPfull[hp_mask], slum[hp_mask]):.3f}")
 
 
# 9. Write GeoTIFF + per-cell CSV
grid = np.full((H, W), -9999.0, dtype="float32")
grid[rows[complete], cols[complete]] = MHI_n[complete]
PROF.update(dtype="float32", nodata=-9999.0, count=1)
with rasterio.open(os.path.join(OUT, "chennai_mhi_100m.tif"), "w", **PROF) as d:
    d.write(grid, 1)
 
pd.DataFrame({"x": xs, "y": ys, "heat": hN, "flood": fN, "poll": pN,
              "MHI": MHI_n, "slum": slum, "complete": complete}) \
  .to_csv(os.path.join(OUT, "chennai_mhi_cells.csv"), index=False)
print("\nwrote chennai_mhi_100m.tif + chennai_mhi_cells.csv")
 
 
# 10. Maps & sensitivity figures
disp = np.where(grid == -9999.0, np.nan, grid)
ext = [T.c, T.c + W*T.a, T.f + H*T.e, T.f]
 
# pure MHI 
fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(disp, extent=ext, origin="upper", cmap="magma_r", vmin=0, vmax=1)
ax.set_title("Composite multi-hazard index (MHI, equal weights, 0-1)")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MHI")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "map_mhi_index.png"), dpi=150); plt.close()
 
# MHI + slum overlay 
fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(disp, extent=ext, origin="upper", cmap="magma_r", vmin=0, vmax=1)
slum_gdf.boundary.plot(ax=ax, color="cyan", linewidth=0.4)
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_title("MHI with slum settlements (cyan) overlaid")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MHI")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "map_mhi_slum_overlay.png"), dpi=150); plt.close()
 
# slum vs non-slum boxplot 
fig, ax = plt.subplots(figsize=(8, 5))
data, labels = [], []
for name, arr in [("Heat", hN), ("Flood", fN), ("Pollution", pN), ("MHI", MHI_n)]:
    data += [arr[complete & slum], arr[complete & ~slum]]
    labels += [f"{name}\nslum", f"{name}\nnon-slum"]
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor("#d1495b" if i % 2 == 0 else "#8ea4bf")
ax.set_ylabel("Normalised hazard score (0-1)")
ax.set_title("Slum vs non-slum: per-hazard layers and composite MHI")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_slum_boxplot.png"), dpi=150); plt.close()
 
# Dirichlet AUC histogram 
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(aucs, bins=30, color="#5b8a72", edgecolor="white")
ax.axvline(0.5, color="k", ls="--", lw=1, label="AUC=0.5 (no difference)")
ax.axvline(aucs.mean(), color="#d1495b", lw=2, label=f"mean={aucs.mean():.3f}")
ax.set_xlabel("Slum vs non-slum AUC on MHI")
ax.set_ylabel("Count (of 500 weightings)")
ax.set_title("Robustness of slum excess under 500 random (Dirichlet) weightings")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_dirichlet_auc.png"), dpi=150); plt.close()
 
print("wrote 4 figures to", OUT)
print("\nDONE.")
 
 


#Level-2 Composite Risk Index (CRI)
import os, numpy as np, pandas as pd, rasterio, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rasterio.warp import transform
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.stats import rankdata, mannwhitneyu
 
PROJ = r"C:\毕设项目"
HEAT  = os.path.join(PROJ, r"data\heat\chennai_heatsusc_100m.tif")
FLOOD = os.path.join(PROJ, r"data\flood\chennai_floodhazard_100m.tif")
POLL  = os.path.join(PROJ, r"data\pollution\chennai_pollution_index_100m.tif")
POP   = os.path.join(PROJ, r"data\composite\chennai_pop_2020_100m.tif")
FTAB  = os.path.join(PROJ, r"data\flood\chennai_flood_variables_tagged.csv")  # impervious, elevation
SLUM  = os.path.join(PROJ, r"data\slum_boundary_shapefile\output.shp")
OUT   = os.path.join(PROJ, r"data\composite")
np.random.seed(42)
 

# 1. Master grid centroids + slum tag
with rasterio.open(HEAT) as s:
    T, W, H, NOD, CRS, PROF = s.transform, s.width, s.height, s.nodata, s.crs, s.profile
    ha = s.read(1).astype("float32")
valid = np.isfinite(ha) & (ha != NOD)
rows, cols = np.where(valid)
xs, ys = rasterio.transform.xy(T, rows, cols); xs, ys = np.array(xs), np.array(ys)
lon, lat = transform("EPSG:32644", "EPSG:4326", xs, ys); lon, lat = np.array(lon), np.array(lat)
 
slum_gdf = gpd.read_file(SLUM).to_crs(CRS)
pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=CRS)
j = gpd.sjoin(pts, slum_gdf[["geometry"]], how="left", predicate="within")
slum = (~j.index_right.isna()).groupby(j.index).max().reindex(range(len(xs))).fillna(False).values.astype(bool)
 
def samp(path):
    with rasterio.open(path) as s:
        arr = s.read(1).astype("float32"); tt, nd, w, h = s.transform, s.nodata, s.width, s.height
    inv = ~tt; c = inv * (xs, ys); cc = np.floor(c[0]).astype(int); rr = np.floor(c[1]).astype(int)
    out = np.full(xs.shape, np.nan, "float32"); ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    out[ok] = arr[rr[ok], cc[ok]]; out[out == nd] = np.nan; return out
 
heat  = samp(HEAT); flood = samp(FLOOD); poll = samp(POLL); pop = samp(POP)
 
# impervious + elevation from flood table -> nearest master cell
ft = pd.read_csv(FTAB).dropna(subset=["lon", "lat", "impervious", "elevation"]).reset_index(drop=True)
d, idx = cKDTree(np.c_[ft.lon.values, ft.lat.values]).query(np.c_[lon, lat], k=1)
tol = 0.0009  # ~100 m in degrees
imp  = np.where(d < tol, ft.impervious.values[idx], np.nan)
elev = np.where(d < tol, ft.elevation.values[idx], np.nan)
 

# 2. Helpers + build the three factors
def mm(a, mask): lo, hi = np.nanmin(a[mask]), np.nanmax(a[mask]); return (a - lo) / (hi - lo)
def auc(score, mask):
    sc, lb = score[mask], slum[mask]; r = rankdata(sc); n1, n0 = lb.sum(), (~lb).sum()
    return (r[lb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
 
comp = np.isfinite(heat) & np.isfinite(flood) & np.isfinite(poll)
MHI  = mm((mm(heat, comp) + mm(flood, comp) + mm(poll, comp)) / 3, comp)   # Level 1
 
vimask = np.isfinite(imp) & np.isfinite(elev)
imp_n, lowelev_n = mm(imp, vimask), mm(-elev, vimask)   # low elevation -> higher vulnerability
VI  = mm((imp_n + lowelev_n) / 2, vimask)
Exp = mm(pop, np.isfinite(pop))
 
cri_mask = comp & np.isfinite(Exp) & vimask
CRI = mm(MHI * Exp * VI, cri_mask)                       # Level 2 (multiplicative)
print(f"CRI cells: {cri_mask.sum()} (slum {slum[cri_mask].sum()})")
 

# 3. Slum vs non-slum, with VI decomposition
def rep(name, score):
    sm, ns = score[cri_mask & slum], score[cri_mask & ~slum]
    p = mannwhitneyu(sm, ns, alternative="two-sided")[1]
    print(f"  {name:18s} slum={sm.mean():.3f} nonslum={ns.mean():.3f} AUC={auc(score, cri_mask):.3f} p={p:.1e}")
print("=== slum vs non-slum (CRI cells) ===")
rep("MHI (hazard)",        MHI)
rep("Exposure (pop)",      Exp)
rep("  impervious only",   mm(imp,  vimask))
rep("  low-elevation only",mm(-elev, vimask))
rep("VI (imp+lowelev)",    VI)
rep("CRI = MHI*E*VI",      CRI)
 
def hot(score, pct):
    thr = np.percentile(score[cri_mask], 100 - pct); h = cri_mask & (score >= thr)
    s = (h & slum).sum() / (cri_mask & slum).sum(); n = (h & ~slum).sum() / (cri_mask & ~slum).sum()
    return s, n, s / n
print("\n=== CRI hotspot over-exposure ===")
for pct in (10, 20, 30):
    s, n, r = hot(CRI, pct); print(f"  top-{pct}%: slum {100*s:.1f}% nonslum {100*n:.1f}% = {r:.2f}x")
 

# 4. Sensitivity: multiplicative vs additive, and VI-weight sweep
print("\n=== sensitivity (slum AUC on CRI) ===")
print(f"  multiplicative MHI*E*VI : {auc(CRI, cri_mask):.3f}")
print(f"  additive (MHI+E+VI)/3   : {auc(mm((MHI+Exp+VI)/3, cri_mask), cri_mask):.3f}")
print(f"  hazard-only MHI         : {auc(MHI, cri_mask):.3f}")
for wv in (0.5, 1.0, 2.0):   # emphasise VI more/less in a geometric form
    sc = mm(MHI * Exp * (VI ** wv), cri_mask)
    print(f"  MHI*E*VI^{wv:<3}          : {auc(sc, cri_mask):.3f}")


# 5. Write GeoTIFF + maps
def write_tif(vals, mask, path):
    g = np.full((H, W), -9999.0, "float32"); g[rows[mask], cols[mask]] = vals[mask]
    PROF.update(dtype="float32", nodata=-9999.0, count=1)
    with rasterio.open(path, "w", **PROF) as d: d.write(g, 1)
    return g
gC = write_tif(CRI, cri_mask, os.path.join(OUT, "chennai_cri_100m.tif"))
write_tif(VI, vimask, os.path.join(OUT, "chennai_vi_100m.tif"))
 
disp = np.where(gC == -9999.0, np.nan, gC)
ext = [T.c, T.c + W*T.a, T.f + H*T.e, T.f]
fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(disp, extent=ext, origin="upper", cmap="inferno_r", vmin=0, vmax=1)
slum_gdf.boundary.plot(ax=ax, color="cyan", linewidth=0.4)
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_title("Composite risk index (CRI = MHI x Exposure x Vulnerability)")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CRI (0-1)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "map_cri_slum_overlay.png"), dpi=150); plt.close()
 
pd.DataFrame({"x": xs, "y": ys, "MHI": MHI, "Exp": Exp, "VI": VI, "CRI": CRI,
              "slum": slum, "cri_valid": cri_mask}).to_csv(
    os.path.join(OUT, "chennai_cri_cells.csv"), index=False)
print("\nwrote chennai_cri_100m.tif, chennai_vi_100m.tif, map_cri_slum_overlay.png, chennai_cri_cells.csv")
 
