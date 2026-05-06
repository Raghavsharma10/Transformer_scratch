def set_data(self, vol, clim=None):
        """ Set the volume data. 

        Parameters
        ----------
        vol : ndarray
            The 3D volume.
        clim : tuple | None
            Colormap limits to use. None will use the min and max values.
        """
        # Check volume
        if not isinstance(vol, np.ndarray):
            raise ValueError('Volume visual needs a numpy array.')
        if not ((vol.ndim == 3) or (vol.ndim == 4 and vol.shape[-1] <= 4)):
            raise ValueError('Volume visual needs a 3D image.')
        
        # Handle clim
        if clim is not None:
            clim = np.array(clim, float)
            if not (clim.ndim == 1 and clim.size == 2):
                raise ValueError('clim must be a 2-element array-like')
            self._clim = tuple(clim)
        if self._clim is None:
            self._clim = vol.min(), vol.max()
        
        # Apply clim
        vol = np.array(vol, dtype='float32', copy=False)
        if self._clim[1] == self._clim[0]:
            if self._clim[0] != 0.:
                vol *= 1.0 / self._clim[0]
        else:
            vol -= self._clim[0]
            vol /= self._clim[1] - self._clim[0]
        
        # Apply to texture
        self._tex.set_data(vol)  # will be efficient if vol is same shape
        self.shared_program['u_shape'] = (vol.shape[2], vol.shape[1], 
                                          vol.shape[0])
        
        shape = vol.shape[:3]
        if self._vol_shape != shape:
            self._vol_shape = shape
            self._need_vertex_update = True
        self._vol_shape = shape
        
        # Get some stats
        self._kb_for_texture = np.prod(self._vol_shape) / 1024