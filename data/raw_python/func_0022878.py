def set_polygon_offset(self, factor=0., units=0.):
        """Set the scale and units used to calculate depth values
    
        Parameters
        ----------
        factor : float
            Scale factor used to create a variable depth offset for
            each polygon.
        units : float
            Multiplied by an implementation-specific value to create a
            constant depth offset.
        """
        self.glir.command('FUNC', 'glPolygonOffset', float(factor),
                          float(units))