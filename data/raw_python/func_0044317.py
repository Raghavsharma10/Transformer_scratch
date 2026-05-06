def _anchor_path(self, anchor_id):
        "Absolute path to the data file for `anchor_id`."
        file_name = '{}.yml'.format(anchor_id)
        file_path = self._spor_dir / file_name
        return file_path