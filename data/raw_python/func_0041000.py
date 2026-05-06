def get_cached_filename(self, filename, extention, settings_list=None):
        """Creates a filename with md5 cache string based on settings list

        Args:
            filename (str): the filename without extention
            extention (str): the file extention without dot. (i.e. 'pkl')
            settings_list (dict|list): the settings list as list (optional)
                NB! The dictionaries have to be sorted or hash id will change
                arbitrarely.
        """
        cached_name = "_".join([filename, self.get_hash()])
        return ".".join([cached_name, extention])