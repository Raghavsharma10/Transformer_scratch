def filenames(self) -> Tuple[str, ...]:
        """A |tuple| of names of all handled |NetCDFFile| objects."""
        return tuple(sorted(set(itertools.chain(
            *(_.keys() for _ in self.folders.values())))))