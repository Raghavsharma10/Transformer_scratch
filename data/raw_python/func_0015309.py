def get_full_dir_name(self):
        """
        Function returns a full dir name
        """
        return os.path.join(self.dir_name.get_text(), self.entry_project_name.get_text())