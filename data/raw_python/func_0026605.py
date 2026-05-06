def map_E_to_height(self, alat, alon, height, newheight, E):
        """Performs mapping of electric field along the magnetic field.

        It is assumed that the electric field is perpendicular to B.

        Parameters
        ==========
        alat : (N,) array_like or float
            Modified apex latitude
        alon : (N,) array_like or float
            Modified apex longitude
        height : (N,) array_like or float
            Source altitude in km
        newheight : (N,) array_like or float
            Destination altitude in km
        E : (3,) or (3, N) array_like
            Electric field (at `alat`, `alon`, `height`) in geodetic east,
            north, and up components

        Returns
        =======
        E : (3, N) or (3,) ndarray
            The electric field at `newheight` (geodetic east, north, and up
            components)

        """

        return self._map_EV_to_height(alat, alon, height, newheight, E, 'E')