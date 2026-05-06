def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        mask = self._data['geometry']['mask']
        if os.path.abspath(mask):
            mask = os.path.split(mask)[-1]
        return mask