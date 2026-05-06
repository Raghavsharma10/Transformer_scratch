def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        try:
            maskid = self._data['maskname']
            if not maskid.endswith('.mat'):
                maskid = maskid + '.mat'
            return maskid
        except KeyError:
            return None