def subdevicenames(self) -> Tuple[str, ...]:
        """A |tuple| containing the device names."""
        self: NetCDFVariableBase
        return tuple(self.sequences.keys())