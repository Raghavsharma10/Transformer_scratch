def set_value(self, key, value):
        """
        Set key value to the file.

        The fuction will be make the key and value to dictinary formate.
        If its exist then it will update the current new key value to
        the file.
        Arg:
        key : cache key
        value : cache value
        """
        file_cache = self.read_file()
        if file_cache:
            file_cache[key] = value
        else:
            file_cache = {}
        file_cache[key] = value
        self.update_file(file_cache)