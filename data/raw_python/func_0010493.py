def from_object(self, obj: Union[str, Any]) -> None:
        """Load values from an object."""
        if isinstance(obj, str):
            obj = importer.import_object_str(obj)

        for key in dir(obj):
            if key.isupper():
                value = getattr(obj, key)
                self._setattr(key, value)

        logger.info("Config is loaded from object: %r", obj)