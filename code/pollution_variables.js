
var MASTER_CRS='EPSG:32644', MASTER_SCALE=100;
var aoi=ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME','Chennai')).geometry();
Map.centerObject(aoi,11);
var Y0=2019, Y1=2024;
function to100(img){ return img.resample('bilinear').reproject({crs:MASTER_CRS, scale:100}); }
 
// main: Sentinel-5P NO2 
var no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
  .select('tropospheric_NO2_column_number_density')
  .filterBounds(aoi).filter(ee.Filter.calendarRange(Y0,Y1,'year'))
  .mean().multiply(1e6);                    
var no2_100 = to100(no2).rename('NO2');
 
// PM reference: MAIAC AOD 1km (Optical_Depth_047, multi-year mean)
var aod = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
  .select('Optical_Depth_047')
  .filterBounds(aoi).filter(ee.Filter.calendarRange(Y0,Y1,'year'))
  .mean().multiply(0.001);
var aod_100 = to100(aod).rename('AOD');

// PM2.5 reference layer (ACAG/WashU annual product).
var pm_acag = ee.Image(ee.ImageCollection('projects/sat-io/open-datasets/GLOBAL-SATELLITE-PM25/ANNUAL')
  .filter(ee.Filter.date('2019-01-01','2019-12-31')).first()).select(0);
var pm25_100 = to100(pm_acag).rename('PM25acag');
 
// downscaling modifier variables
// built-up 
var builtup100 = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface').divide(10000).resample('bilinear')
  .reproject({crs:MASTER_CRS, scale:100}).rename('builtup');
 
// NDVI + NDBI 
function maskL8(img){
  var qa=img.select('QA_PIXEL');
  var m=qa.bitwiseAnd(1<<1).eq(0).and(qa.bitwiseAnd(1<<3).eq(0)).and(qa.bitwiseAnd(1<<4).eq(0));
  return img.updateMask(m).updateMask(img.select('QA_RADSAT').eq(0));
}
var l8=ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(aoi)
  .filter(ee.Filter.calendarRange(Y0,Y1,'year')).map(maskL8);
var sr=l8.map(function(i){return i.select('SR_B.').multiply(0.0000275).add(-0.2)
  .copyProperties(i,['system:time_start']);});
var ndvi=sr.map(function(i){return i.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');}).median();
var ndbi=sr.map(function(i){return i.normalizedDifference(['SR_B6','SR_B5']).rename('NDBI');}).median();
var nativeL8=ee.Image(l8.first()).select('SR_B4').projection();
function agg(img){ return img.setDefaultProjection(nativeL8)
  .reduceResolution({reducer:ee.Reducer.mean(),maxPixels:1024}).reproject({crs:MASTER_CRS,scale:100}); }
var ndvi100=agg(ndvi).rename('NDVI');
var ndbi100=agg(ndbi).rename('NDBI');
 
// VIIRS
var viirs=ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .select('avg_rad').filterBounds(aoi).filter(ee.Filter.calendarRange(Y0,Y1,'year')).mean();
var viirs100=to100(viirs).rename('VIIRS');
 
// // stack bands + sample and export CSV
var stack=no2_100.addBands(aod_100).addBands(pm25_100).addBands(builtup100)
  .addBands(ndvi100).addBands(ndbi100).addBands(viirs100).clip(aoi);
 
Map.addLayer(no2_100,{min:20,max:120,palette:['green','yellow','red']},'NO2 (µmol/m2)',false);
Map.addLayer(viirs100,{min:0,max:60,palette:['black','yellow','white']},'VIIRS',false);
 
var samples=stack.sample({region:aoi, scale:100, projection:MASTER_CRS,
  factor:1, geometries:true, dropNulls:true, tileScale:4});
print('Sample count 19567):', samples.size());
print('The first Preview:', samples.first());
Export.table.toDrive({collection:samples, description:'chennai_pollution_variables_table',
  folder:'chennai_pollution', fileFormat:'CSV'});
