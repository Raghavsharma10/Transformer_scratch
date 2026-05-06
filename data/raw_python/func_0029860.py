def list_records(self, file_const=None):
        """Iterate through the file records"""
        for r in self._dataset.files:
            if file_const and r.minor_type != file_const:
                continue
            yield self.instance_from_name(r.path)