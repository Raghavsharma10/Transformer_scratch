def get_files(self):
        """stub"""
        files_map = {}
        try:
            files_map['choices'] = self.get_choices_file_urls_map()
            try:
                files_map.update(self.get_file_urls_map())
            except IllegalState:
                pass
        except Exception:
            files_map['choices'] = self.get_choices_files_map()
            try:
                files_map.update(self.get_files_map())
            except IllegalState:
                pass
        return files_map