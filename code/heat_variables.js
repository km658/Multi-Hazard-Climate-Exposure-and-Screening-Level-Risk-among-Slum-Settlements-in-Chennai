var MASTER_CRS   = 'EPSG:32644';   // UTM zone 44N, cover Chennai
var MASTER_SCALE = 100;            
var START_YEAR = 2019, END_YEAR = 2024;
var HOT_START = 3, HOT_END = 6;   

//  AOI: Chennai boundary
var chennai = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME', 'Chennai'));
var aoi = chennai.geometry();
Map.centerObject(aoi, 10);

// Landsat 8 C2 L2 
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');

// cloud / cloud-shadow / saturation mask
function maskL8(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))       // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))       // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0));      // cloud shadow
  var sat = image.select('QA_RADSAT').eq(0);
  return image.updateMask(mask).updateMask(sat);
}

// add scaling + compute LST (°C) and NDVI
function scaleAndIndex(image) {
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var lstC = image.select('ST_B10').multiply(0.00341802).add(149.0)
                  .subtract(273.15).rename('LST');
  var ndvi = optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
// Liang (2001) broadband shortwave albedo (ETM+ coefficients approximated for OLI)
  var albedo = optical.expression(
    '(0.356*B + 0.130*R + 0.373*NIR + 0.085*SWIR1 + 0.072*SWIR2 - 0.0018) / 1.016', {
      'B':     optical.select('SR_B2'),
      'R':     optical.select('SR_B4'),
      'NIR':   optical.select('SR_B5'),
      'SWIR1': optical.select('SR_B6'),
      'SWIR2': optical.select('SR_B7')
  }).rename('albedo');
  var ndbi = optical.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI');
  return image.addBands(lstC).addBands(ndvi).addBands(albedo).addBands(ndbi);
 
  
}

var summer = l8
  .filterBounds(aoi)
  .filter(ee.Filter.calendarRange(START_YEAR, END_YEAR, 'year'))
  .filter(ee.Filter.calendarRange(HOT_START, HOT_END, 'month'))
  .map(maskL8)
  .map(scaleAndIndex);
print('number of images used for composite:', summer.size());

var lst = summer.select('LST').median();
var ndvi = summer.select('NDVI').median();

// Aggregate to the 100m master grid
function to100m(img) {
  return img
    .reproject({crs: MASTER_CRS, scale: 30})   
    .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
    .reproject({crs: MASTER_CRS, scale: 100});  
}
var lst100  = to100m(lst).clip(aoi).rename('LST');
var ndvi100 = to100m(ndvi).clip(aoi).rename('NDVI');

var albedo = summer.select('albedo').median();
var albedo100 = to100m(albedo).clip(aoi).rename('albedo');

Map.addLayer(albedo100, {min: 0.05, max: 0.30,
  palette: ['black','white']}, 'albedo');
print('albedo min/max:', albedo100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi,
  scale: 200, bestEffort: true, maxPixels: 1e9}));

// check
Map.addLayer(lst100, {min: 28, max: 45,
  palette: ['blue','yellow','red']}, 'LST (°C)');
Map.addLayer(ndvi100, {min: 0, max: 0.6,
  palette: ['white','green']}, 'NDVI');
print('LST min/max (°C):', lst100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi, scale: 100, maxPixels: 1e9}));

// Export
Export.image.toDrive({
  image: lst100.toFloat(), description: 'chennai_LST_summer_100m',
  folder: 'chennai_heat', region: aoi,
  scale: MASTER_SCALE, crs: MASTER_CRS, maxPixels: 1e9});
Export.image.toDrive({
  image: ndvi100.toFloat(), description: 'chennai_NDVI_summer_100m',
  folder: 'chennai_heat', region: aoi,
  scale: MASTER_SCALE, crs: MASTER_CRS, maxPixels: 1e9});
  
  
  
  // built-up
var builtup100 = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .divide(10000)                                
  .resample('bilinear')
  .reproject({crs: MASTER_CRS, scale: 100})
  .clip(aoi).rename('builtup');

Map.addLayer(builtup100, {min: 0, max: 1,
  palette: ['white','red']}, 'built-up');
print('built-up min/max:', builtup100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi,
  scale: 200, bestEffort: true, maxPixels: 1e9}));
  
  
  //distance to water body / coast 
var waterMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence').gte(50)          
  .unmask(0)
  .reproject({crs: MASTER_CRS, scale: 100});

var distWater100 = waterMask
  .fastDistanceTransform(1024).sqrt()    
  .multiply(100)                        
  .reproject({crs: MASTER_CRS, scale: 100})
  .clip(aoi).rename('dist_water');

Map.addLayer(distWater100, {min: 0, max: 15000,
  palette: ['blue','white','red']}, 'dist to water (m)');
print('dist_water min/max (m):', distWater100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi,
  scale: 200, bestEffort: true, maxPixels: 1e9}));
  
  // NDBI impervious/built-up
var ndbi = summer.select('NDBI').median();
var ndbi100 = to100m(ndbi).clip(aoi).rename('NDBI');

Map.addLayer(ndbi100, {min: -0.2, max: 0.3,
  palette: ['green','white','brown']}, 'NDBI');
print('NDBI min/max:', ndbi100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi,
  scale: 200, bestEffort: true, maxPixels: 1e9}));
  
  
  // bare-soil fraction 
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var bare100 = wc.eq(60)                         
  .reproject({crs: MASTER_CRS, scale: 10})       
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: MASTER_CRS, scale: 100})      
  .clip(aoi).rename('bare');

Map.addLayer(bare100, {min: 0, max: 0.5,
  palette: ['green','yellow','brown']}, 'bare soil');
print('bare min/max:', bare100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi,
  scale: 200, bestEffort: true, maxPixels: 1e9}));
print('bare mean:', bare100.reduceRegion({reducer: ee.Reducer.mean(),
  geometry: aoi, scale: 200, bestEffort: true, maxPixels: 1e9}));
  

print('NDVI min/max:', ndvi100.reduceRegion({reducer: ee.Reducer.minMax(),
  geometry: aoi, scale: 200, bestEffort: true, maxPixels: 1e9}));
  
  
  // building height / SVF proxy
var bheight100 = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_H/2018')
  .select('built_height')
  .resample('bilinear')
  .reproject({crs: MASTER_CRS, scale: 100})
  .clip(aoi).rename('bheight');

Map.addLayer(bheight100, {min: 0, max: 15,
  palette: ['white','purple']}, 'building height');
print('bheight min/max (m):', bheight100.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: aoi,
  scale: 200, bestEffort: true, maxPixels: 1e9}));

// Stack all heat candidate variables and export the 100 m cell table used by the Python analysis.
var stack = lst100
  .addBands(ndbi100)
  .addBands(ndvi100)
  .addBands(albedo100)
  .addBands(bare100)
  .addBands(bheight100)
  .addBands(builtup100)
  .addBands(distWater100)
  .clip(aoi);

var samples = stack.sample({
  region: aoi,
  scale: MASTER_SCALE,
  projection: MASTER_CRS,
  factor: 1,
  geometries: true,
  dropNulls: true,
  tileScale: 4
});

print('Sample count:', samples.size());
print('First sample preview:', samples.first());

Export.table.toDrive({
  collection: samples,
  description: 'chennai_heat_variables_table',
  folder: 'chennai_heat',
  fileFormat: 'CSV'
});
