def update(self):
        """Update |RelSoilArea| based on |Area|, |ZoneArea|, and |ZoneType|.

        >>> from hydpy.models.hland import *
        >>> parameterstep('1d')
        >>> nmbzones(4)
        >>> zonetype(FIELD, FOREST, GLACIER, ILAKE)
        >>> area(100.0)
        >>> zonearea(10.0, 20.0, 30.0, 40.0)
        >>> derived.relsoilarea.update()
        >>> derived.relsoilarea
        relsoilarea(0.3)
        """
        con = self.subpars.pars.control
        temp = con.zonearea.values.copy()
        temp[con.zonetype.values == GLACIER] = 0.
        temp[con.zonetype.values == ILAKE] = 0.
        self(numpy.sum(temp)/con.area)