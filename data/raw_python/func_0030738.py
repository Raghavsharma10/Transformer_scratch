def root_dataset(self):
        """Return the root dataset, which hold configuration values for the library"""
        ds = self.dataset(ROOT_CONFIG_NAME_V)
        ds._database = self
        return ds