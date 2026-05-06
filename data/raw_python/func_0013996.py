def _remove_data_dir_path(self, inp=None):
        # import string
        """Remove the data directory path from filenames"""
        # need to add a check in here to make sure data_dir path is actually in
        # the filename
        if inp is not None:
            split_str = os.path.join(self.data_path, '')
            return inp.apply(lambda x: x.split(split_str)[-1])