# Multi-Hazard-Climate-Exposure-and-Screening-Level-Risk-among-Slum-Settlements-in-Chennai
Code and reproducibility materials for the MSc Data Science Extended Research Project.


The workflow combines Google Earth Engine exports, local Python processing, and processed geospatial inputs. Full source dataset links are provided in the accompanying additional-material submission document rather than repeated here.

## Structure


code/
  heat_variables.js
  flood_variables.js
  pollution_variables.js
  analysis_pipeline.py
  extract_population.py

input/
  chennai_heat_variables_table.csv
  chennai_heat_variables_tagged.csv
  chennai_flood_variables_table.csv
  chennai_flood_variables_tagged.csv
  chennai_pollution_variables_table.xlsx
  chennai_pop_2020_100m.tif
  osm_road_industry.csv
  slum_boundary_shapefile/


## Environment

The Python scripts were run in a Jupyter Python environment using Python 3.10.19 (Anaconda, 64-bit). The main package versions used are:


pandas==2.3.3
numpy==1.26.4
matplotlib==3.10.7
scipy==1.15.3
pingouin==0.6.1
geopandas==1.1.1
shapely==2.0.7
rasterio==1.4.4
pyproj==3.7.1
xarray==2025.6.1
h5netcdf==1.6.1
openpyxl==3.1.5

## Input Data

The `input/` folder contains the processed inputs needed to reproduce the main analysis outputs without re-downloading the largest raw files.

Large raw input files are not included in this repository due to file-size limits. These include the national WorldPop raster, the ACAG/TROPOMI surface NO2 NetCDF file, and the BBBike/OpenStreetMap GeoPackage extract. Source links for these datasets are listed in the accompanying additional-material submission document.

The Google Earth Engine scripts retrieve public GEE datasets directly and export the 100 m grid-level variable tables used by the Python workflow:


heat_variables.js       -> chennai_heat_variables_table.csv
flood_variables.js      -> chennai_flood_variables_table.csv
pollution_variables.js  -> chennai_pollution_variables_table.csv / .xlsx


## Running the Workflow

The scripts were developed using the local project path `C:\毕设项目`. To run them elsewhere, update the path constants at the top of the Python scripts, or recreate the same folder structure.

Recommended order:

1. Run the three Google Earth Engine scripts in `code/` if regenerating the GEE variable tables.
2. Place the exported tables and processed input files in the expected data folders.
3. Run `code/analysis_pipeline.py` to generate the single-hazard screening outputs, hazard maps, composite MHI outputs, and figures.
4. Run `code/extract_population.py` before the CRI stage if regenerating `chennai_pop_2020_100m.tif` from the original WorldPop raster.
5. Re-run the CRI section of `analysis_pipeline.py` after the population raster has been generated.

## Main Dissertation Outputs

The workflow generates the mapped and statistical outputs used as dissertation figures, including:


spearman_matrix.png
flood_spearman_matrix.png
pollution_spearman_matrix.png
map_heat_susc_index.png
map_heat_susc_slum_overlay.png
map_flood_hazard_slum_overlay.png
map_pollution_slum_overlay.png
map_mhi_index.png
map_mhi_slum_overlay.png
fig_slum_boxplot.png
fig_dirichlet_auc.png


It also writes aligned raster and tabular outputs such as:


chennai_heatsusc_100m.tif
chennai_floodhazard_100m.tif
chennai_pollution_index_100m.tif
chennai_mhi_100m.tif
chennai_cri_100m.tif
chennai_mhi_cells.csv
chennai_cri_cells.csv


## Notes

The repository is intended to support reproducibility of the submitted analysis. The processed inputs are included where practical; very large raw source files are documented through external source links in the additional-material submission document.
