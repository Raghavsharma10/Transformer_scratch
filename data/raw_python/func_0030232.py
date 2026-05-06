def geoframe(self, sql, simplify=None, crs=None, epsg=4326):
        """
        Return geopandas dataframe

        :param simplify: Integer or None. Simplify the geometry to a tolerance, in the units of the geometry.
        :param crs: Coordinate reference system information
        :param epsg: Specifiy the CRS as an EPGS number.
        :return: A Geopandas GeoDataFrame
        """
        import geopandas
        from shapely.wkt import loads
        from fiona.crs import from_epsg

        if crs is None:
            try:
                crs = from_epsg(epsg)
            except TypeError:
                raise TypeError('Must set either crs or epsg for output.')

        df = self.dataframe(sql)
        geometry = df['geometry']

        if simplify:
            s = geometry.apply(lambda x: loads(x).simplify(simplify))
        else:
            s = geometry.apply(lambda x: loads(x))

        df['geometry'] = geopandas.GeoSeries(s)

        return geopandas.GeoDataFrame(df, crs=crs, geometry='geometry')