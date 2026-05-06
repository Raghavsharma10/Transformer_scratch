def _get_init_release_tag(self):
        """
        parses init.py to get previous version
        """
        self.init_version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                      open(self.init_file, "r").read(),
                                      re.M).group(1)