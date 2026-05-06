def get_single_file_info(self, rel_path):
        """ Gets last change time for a single file """

        f_path = self.get_full_file_path(rel_path)
        return get_single_file_info(f_path, rel_path)