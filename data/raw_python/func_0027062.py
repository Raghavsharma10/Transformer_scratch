def loader(self):
        """Create a lazy loader source file loader."""
        loader = super().loader
        if self._lazy and (sys.version_info.major, sys.version_info.minor) != (3, 4):
            loader = LazyLoader.factory(loader)
        # Strip the leading underscore from slots
        return partial(
            loader, **{object.lstrip("_"): getattr(self, object) for object in self.__slots__}
        )