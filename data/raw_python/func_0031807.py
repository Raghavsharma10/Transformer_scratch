def render(self, nb_class=8, disc_func=None, user_defined_breaks=None,
               output="GeoJSON", new_mask=False):
        """
        Parameters
        ----------
        nb_class : int, optionnal
            The number of class (default: 8).
        disc_func : str, optionnal
            The kind of data classification to be used (to be choosed in
            "equal_interval", "jenks", "percentiles, "head_tail_breaks"
            and "prog_geom"), default: None.
        user_defined_breaks : list or tuple, optionnal
            A list of ordered break to use to construct the contours
            (override `nb_class` and `disc_func` values if any)
            (default: None).
        output : string, optionnal
            The type of output expected (not case-sensitive)
            in {"GeoJSON", "GeoDataFrame"} (default: "GeoJSON").
        new_mask : str, optionnal
            Use a new mask by giving the path to the file (Polygons only)
            to use as clipping mask, can also be directly a GeoDataFrame
            (default: False).

        Returns
        -------
        smoothed_result : bytes or GeoDataFrame
            The result, dumped as GeoJSON (utf-8 encoded) or as a GeoDataFrame.
        """
        if disc_func and 'jenks' in disc_func and not jenks_breaks:
            raise ValueError(
                "Missing jenkspy package - could not use jenks breaks")

        zi = self.zi

        if isinstance(new_mask, (type(False), type(None))):
            if not self.use_mask:
                self.use_mask = False
                self.mask = None
        else:
            self.open_mask(new_mask, None)

        # We want levels with the first break value as the minimum of the
        # interpolated values and the last break value as the maximum of theses
        # values:
        if user_defined_breaks:
            levels = user_defined_breaks
            if levels[len(levels) - 1] < np.nanmax(zi):
                levels = levels + [np.nanmax(zi)]
            if levels[0] > np.nanmin(zi):
                levels = [np.nanmin(zi)] + levels
        else:
            levels = self.define_levels(nb_class, disc_func)

        # Ensure that the levels are unique/increasing
        #  to avoid error from `contourf` :
        s_levels = set(levels)
        if len(s_levels) != len(levels):
            levels = list(s_levels)
        levels.sort()

        try:
            collec_poly = contourf(
                self.XI, self.YI,
                zi.reshape(tuple(reversed(self.shape))).T,
                levels,
                vmax=abs(np.nanmax(zi)), vmin=-abs(np.nanmin(zi)))
        # Retry without setting the levels :
        except ValueError:
            collec_poly = contourf(
                self.XI, self.YI,
                zi.reshape(tuple(reversed(self.shape))).T,
                vmax=abs(np.nanmax(zi)), vmin=-abs(np.nanmin(zi)))

        # Fetch the levels returned by contourf:
        levels = collec_poly.levels
        # Set the maximum value at the maximum value of the interpolated values:
        levels[-1] = np.nanmax(zi)
        # Transform contourf contours into a GeoDataFrame of (Multi)Polygons:
        res = isopoly_to_gdf(collec_poly, levels=levels[1:], field_name="max")

        if self.longlat:
            def f(x, y, z=None):
                return (x / 0.017453292519943295,
                        y / 0.017453292519943295)
            res.geometry = [transform(f, g) for g in res.geometry]

        res.crs = self.proj_to_use
        # Set the min/max/center values of each class as properties
        # if this contour layer:
        res["min"] = [np.nanmin(zi)] + res["max"][0:len(res)-1].tolist()
        res["center"] = (res["min"] + res["max"]) / 2

        # Compute the intersection between the contour layer and the mask layer:
        ix_max_ft = len(res) - 1
        if self.use_mask:
            res.loc[0:ix_max_ft, "geometry"] = res.geometry.buffer(
                0).intersection(unary_union(self.mask.geometry.buffer(0)))

        # res.loc[0:ix_max_ft, "geometry"] = res.geometry.buffer(
        #     0).intersection(self.poly_max_extend.buffer(-0.1))

        # Repair geometries if necessary :
        if not all(t in ("MultiPolygon", "Polygon") for t in res.geom_type):
            res.loc[0:ix_max_ft, "geometry"] = \
                [geom if geom.type in ("Polygon", "MultiPolygon")
                 else MultiPolygon(
                     [j for j in geom if j.type in ('Polygon', 'MultiPolygon')]
                     )
                 for geom in res.geometry]

        if "geojson" in output.lower():
            return res.to_crs({"init": "epsg:4326"}).to_json().encode()
        else:
            return res