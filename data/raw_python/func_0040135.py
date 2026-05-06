def _set_linters(self):
        """Use user linters or all available when not specified."""
        if 'linters' in self._config:
            self.user_linters = list(self._parse_cfg_linters())
            self.linters = {linter: self._all_linters[linter]
                            for linter in self.user_linters}
        else:
            self.linters = self._all_linters