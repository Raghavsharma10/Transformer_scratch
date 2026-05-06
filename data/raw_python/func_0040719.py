def _create_output_from_match(self, match_result):
        """As isort outputs full path, we change it to relative path."""
        full_path = match_result['full_path']
        path = self._get_relative_path(full_path)
        return LinterOutput(self.name, path, match_result['msg'])