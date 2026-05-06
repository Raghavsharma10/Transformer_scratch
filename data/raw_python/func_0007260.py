def project_geometry(geometry, crs, to_latlong=False):
    """
    Project a shapely Polygon or MultiPolygon from WGS84 to UTM, or vice-versa

    Parameters
    ----------
    geometry : shapely Polygon or MultiPolygon
        the geometry to project
    crs : int
        the starting coordinate reference system of the passed-in geometry
    to_latlong : bool
        if True, project from crs to WGS84, if False, project
        from crs to local UTM zone

    Returns
    -------
    geometry_proj, crs : tuple (projected shapely geometry, crs of the
    projected geometry)
    """
    gdf = gpd.GeoDataFrame()
    gdf.crs = crs
    gdf.name = 'geometry to project'
    gdf['geometry'] = None
    gdf.loc[0, 'geometry'] = geometry
    gdf_proj = project_gdf(gdf, to_latlong=to_latlong)
    geometry_proj = gdf_proj['geometry'].iloc[0]
    return geometry_proj, gdf_proj.crs