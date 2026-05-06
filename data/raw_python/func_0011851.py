def find_datafile(self, name, search_path=None):
        """
        find all matching data files in search_path
        returns array of tuples (codec_object, filename)
        """
        if not search_path:
            search_path = self.define_dir

        return codec.find_datafile(name, search_path)