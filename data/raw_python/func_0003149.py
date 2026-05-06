def load_scatter_table(self, fn):
        """Load the scattering lookup tables.
        
        Load the scattering lookup tables saved with save_scatter_table.

        Args:
            fn: The name of the scattering table file.            
        """
        data = pickle.load(file(fn))

        if ("version" not in data) or (data["version"]!=tmatrix_aux.VERSION):
            warnings.warn("Loading data saved with another version.", Warning)

        (self.num_points, self.D_max, self._psd_D, self._S_table, 
            self._Z_table, self._angular_table, self._m_table, 
            self.geometries) = data["psd_scatter"]
        return (data["time"], data["description"])