def map_V_to_height(self, alat, alon, height, newheight, V):
        """Performs mapping of electric drift velocity along the magnetic field.

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
        V : (3,) or (3, N) array_like
            Electric drift velocity (at `alat`, `alon`, `height`) in geodetic
            east, north, and up components

        Returns
        =======
        V : (3, N) or (3,) ndarray
            The electric drift velocity at `newheight` (geodetic east, north,
            and up components)

        """

        return self._map_EV_to_height(alat, alon, height, newheight, V, 'V')