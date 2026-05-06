def _save_yaml_file(self, file, val):
        """
        Save data to yaml file

        :param file: Writable object or path to file
        :type file: FileIO | str | unicode
        :param val: Value or struct to save
        :type val: None | int | float | str | unicode | list | dict
        :raises IOError: Failed to save
        """
        try:
            save_yaml_file(file, val)
        except:
            self.exception("Failed to save to {}".format(file))
            raise IOError("Saving file failed")