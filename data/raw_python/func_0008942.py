def mk_geotiff_obj(raster, fn, bands=1, gdal_data_type=gdal.GDT_Float32,
                   lat=[46, 45], lon=[-73, -72]):
    """
    Creates a new geotiff file objects using the WGS84 coordinate system, saves
    it to disk, and returns a handle to the python file object and driver

    Parameters
    ------------
    raster : array
        Numpy array of the raster data to be added to the object
    fn : str
        Name of the geotiff file
    bands : int (optional)
        See :py:func:`gdal.GetDriverByName('Gtiff').Create
    gdal_data : gdal.GDT_<type>
        Gdal data type (see gdal.GDT_...)
    lat : list
        northern lat, southern lat
    lon : list
        [western lon, eastern lon]
    """
    NNi, NNj = raster.shape
    driver = gdal.GetDriverByName('GTiff')
    obj = driver.Create(fn, NNj, NNi, bands, gdal_data_type)
    pixel_height = -np.abs(lat[0] - lat[1]) / (NNi - 1.0)
    pixel_width = np.abs(lon[0] - lon[1]) / (NNj - 1.0)
    obj.SetGeoTransform([lon[0], pixel_width, 0, lat[0], 0, pixel_height])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    obj.SetProjection(srs.ExportToWkt())
    obj.GetRasterBand(1).WriteArray(raster)
    return obj, driver