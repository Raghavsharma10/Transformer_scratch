def from_polygon_skycoord(cls, skycoord, inside=None, max_depth=10):
        """
        Creates a MOC from a polygon.

        The polygon is given as an `astropy.coordinates.SkyCoord` that contains the 
        vertices of the polygon. Concave and convex polygons are accepted but
        self-intersecting ones are currently not properly handled.

        Parameters
        ----------
        skycoord : `astropy.coordinates.SkyCoord`
            The sky coordinates defining the vertices of a polygon. It can describe a convex or
            concave polygon but not a self-intersecting one.
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
        return MOC.from_polygon(lon=skycoord.icrs.ra, lat=skycoord.icrs.dec,
                                inside=inside, max_depth=max_depth)