def _get_relative_path(self, full_path):
        """Return the relative path from current path."""
        try:
            rel_path = Path(full_path).relative_to(Path().absolute())
        except ValueError:
            LOG.error("%s: Couldn't find relative path of '%s' from '%s'.",
                      self.name, full_path, Path().absolute())
            return full_path
        return str(rel_path)