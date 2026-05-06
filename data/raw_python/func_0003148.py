def save_scatter_table(self, fn, description=""):
        """Save the scattering lookup tables.
        
        Save the state of the scattering lookup tables to a file.
        This can be loaded later with load_scatter_table.

        Other variables will not be saved, but this does not matter because
        the results of the computations are based only on the contents
        of the table.

        Args:
           fn: The name of the scattering table file. 
           description (optional): A description of the table.
        """
        data = {
           "description": description,
           "time": datetime.now(),
           "psd_scatter": (self.num_points, self.D_max, self._psd_D, 
                self._S_table, self._Z_table, self._angular_table, 
                self._m_table, self.geometries),
           "version": tmatrix_aux.VERSION
           }
        pickle.dump(data, file(fn, 'w'), pickle.HIGHEST_PROTOCOL)