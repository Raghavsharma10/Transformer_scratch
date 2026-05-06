def set_depth_range(self, near=0., far=1.):
        """Set depth values
    
        Parameters
        ----------
        near : float
            Near clipping plane.
        far : float
            Far clipping plane.
        """
        self.glir.command('FUNC', 'glDepthRange', float(near), float(far))