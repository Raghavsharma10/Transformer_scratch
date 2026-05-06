def record_to_fs(self):
        """Create a filesystem file from a File"""

        fr = self.record

        fn_path = self.file_name

        if fr.contents:
            if six.PY2:
                with self._fs.open(fn_path, 'wb') as f:
                    self.record_to_fh(f)
            else:
                # py3
                with self._fs.open(fn_path, 'w', newline='') as f:
                    self.record_to_fh(f)