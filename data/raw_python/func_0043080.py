def parse(self, requires_cfg=True):
        """Parse the configuration sources into `Bison`.

        Args:
            requires_cfg (bool): Specify whether or not parsing should fail
                if a config file is not found. (default: True)
        """
        self._parse_default()
        self._parse_config(requires_cfg)
        self._parse_env()