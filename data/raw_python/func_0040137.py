def _parse_cfg_linters(self):
        """Return valid linter names found in config files."""
        user_value = self._config.get('linters', '')
        # For each line of "linters" value, use comma as separator
        for line in user_value.splitlines():
            yield from self._parse_linters_line(line)