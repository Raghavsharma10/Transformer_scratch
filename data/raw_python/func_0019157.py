def toys(self) -> Tuple[timetools.TOY, ...]:
        """A sorted |tuple| of all contained |TOY| objects."""
        return tuple(toy for (toy, _) in self)