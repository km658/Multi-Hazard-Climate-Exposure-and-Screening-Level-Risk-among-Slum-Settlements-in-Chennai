var MASTER_CRS = 'EPSG:32644', MASTER_SCALE = 100;
var chennai = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME', 'Chennai'));
var aoi = chennai.geometry();
Map.centerObject(aoi, 11);

// MERIT Hydro 
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var elv  = merit.select('elv');         
var hnd  = merit.select('hnd');         
var upa  = merit.select('upa');          

// slope: computed in MERIT elevation's native projection, then reprojected
var slope = ee.Terrain.slope(elv);       

// TWI = ln( a / tan(beta) )
var slope_rad = slope.multiply(Math.PI/180);
var tanb = slope_rad.tan().max(0.001);
var a_area = upa.multiply(1e6).divide(90);         
var twi = a_area.divide(tanb).log().rename('TWI');

function to100(img){ return img.resample('bilinear').reproject({crs:MASTER_CRS, scale:100}); }
var elv100   = to100(elv).rename('elevation');
var slope100 = to100(slope).rename('slope');
var hnd100   = to100(hnd).rename('HAND');
var upa100   = to100(upa).rename('flow_accum');
var twi100   = to100(twi).rename('TWI');

// Distance to water body (JRC GSW, reusing heat-layer approach; flood: closer to water = more flood-prone)
var waterMask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence').gte(50).unmask(0)
  .reproject({crs:MASTER_CRS, scale:100});
var distWater100 = waterMask.fastDistanceTransform(1024).sqrt().multiply(100)
  .reproject({crs:MASTER_CRS, scale:100}).rename('dist_water');
//

var builtup100 = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface').divide(10000)
  .resample('bilinear').reproject({crs:MASTER_CRS, scale:100}).rename('builtup');

var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var imperv = wc.eq(50).or(wc.eq(60)).rename('impervious');
var imperv100 = imperv.setDefaultProjection(wc.projection())
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs:MASTER_CRS, scale:100}).rename('impervious');

var stack = elv100.addBands(slope100).addBands(hnd100).addBands(upa100)
  .addBands(twi100).addBands(distWater100).addBands(builtup100)
  .addBands(imperv100).clip(aoi);

// skim
Map.addLayer(hnd100, {min:0,max:20,palette:['blue','white','red']}, 'HAND (m)', false);
Map.addLayer(elv100, {min:0,max:30,palette:['blue','green','brown']}, 'elevation (m)', false);
Map.addLayer(twi100, {min:2,max:15,palette:['white','blue']}, 'TWI', false);

var samples = stack.sample({region: aoi, scale: 100, projection: MASTER_CRS,
  factor: 1, geometries: true, dropNulls: true, tileScale: 4});
print('number of sample(19567):', samples.size());
print('preview:', samples.first());

Export.table.toDrive({collection: samples,
  description: 'chennai_flood_variables_table', folder: 'chennai_flood', fileFormat: 'CSV'});
