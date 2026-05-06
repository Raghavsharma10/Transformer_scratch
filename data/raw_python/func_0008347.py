def delete_data(self):
        """
        Delete everything which is related to the plugin. **Do not use if you do not know what you do!**
        """
        self.clean_up()
        tools.delete_dir_rec(self._download_path)
        if self._save_state_file.exists():
            self._save_state_file.unlink()