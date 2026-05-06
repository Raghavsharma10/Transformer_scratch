def record_to_fs(self):
        """Create a filesystem file from a File"""

        fr = self.record

        if fr.contents:
            with self._fs.open(self.file_name, 'w', encoding='utf-8') as f:
                self.record_to_fh(f)