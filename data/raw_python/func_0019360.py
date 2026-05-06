def filepath(self) -> str:
        """The NetCDF file path."""
        return os.path.join(self._dirpath, self.name + '.nc')