def from_polygon(cls, lon, lat, inside=None, max_depth=10):
        """
        Creates a MOC from a polygon

        The polygon is given as lon and lat `astropy.units.Quantity` that define the 
        vertices of the polygon. Concave and convex polygons are accepted but
        self-intersecting ones are currently not properly handled.

        Parameters
        ----------
        lon : `astropy.units.Quantity`
            The longitudes defining the polygon. Can describe convex and
            concave polygons but not self-intersecting ones.
        lat : `astropy.units.Quantity`
            The latitudes defining the polygon. Can describe convex and concave
            polygons but not self-intersecting ones.
        inside : `astropy.coordinates.SkyCoord`, optional
            A point that will be inside the MOC is needed as it is not possible to determine the inside area of a polygon 
            on the unit sphere (there is no infinite area that can be considered as the outside because on the sphere,
            a closed polygon delimits two finite areas).
            Possible improvement: take the inside area as the one covering the smallest region on the sphere.

            If inside=None (default behavior), the mean of all the vertices is taken as lying inside the polygon. That approach may not work for 
            concave polygons.
        max_depth : int, optional
            The resolution of the MOC. Set to 10 by default.

        Returns
        -------
        result : `~mocpy.moc.MOC`
            The resulting MOC
        """
        from .polygon import PolygonComputer

        polygon_computer = PolygonComputer(lon, lat, inside, max_depth)
        # Create the moc from the python dictionary

        moc = MOC.from_json(polygon_computer.ipix)
        # We degrade it to the user-requested order
        if polygon_computer.degrade_to_max_depth:
            moc = moc.degrade_to_order(max_depth)

        return moc