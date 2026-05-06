def from_pyfile(self, filename: str) -> None:
        """Load values from a Python file."""
        globals_ = {}  # type: Dict[str, Any]
        locals_ = {}  # type: Dict[str, Any]
        with open(filename, "rb") as f:
            exec(compile(f.read(), filename, 'exec'), globals_, locals_)

        for key, value in locals_.items():
            if (key.isupper() and not isinstance(value, types.ModuleType)):
                self._setattr(key, value)

        logger.info("Config is loaded from file: %s", filename)