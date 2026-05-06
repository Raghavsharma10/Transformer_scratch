def _cleanup(self, domains):
        """Remove the temporary '.pot' files that were created for the domains."""
        for option in domains.values():
            try:
                os.remove(option['pot'])
            except (IOError, OSError):
                # It is not a problem if we can't actually remove the temporary file
                pass