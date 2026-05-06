def _save_json_file(
        self, file, val,
        pretty=False, compact=True, sort=True, encoder=None
    ):
        """
        Save data to json file

        :param file: Writable file or path to file
        :type file: FileIO | str | unicode
        :param val: Value or struct to save
        :type val: None | int | float | str | list | dict
        :param pretty: Format data to be readable (default: False)
        :type pretty: bool
        :param compact: Format data to be compact (default: True)
        :type compact: bool
        :param sort: Sort keys (default: True)
        :type sort: bool
        :param encoder: Use custom json encoder
        :type encoder: T <= flotils.loadable.DateTimeEncoder
        :rtype: None
        :raises IOError: Failed to save
        """
        try:
            save_json_file(file, val, pretty, compact, sort, encoder)
        except:
            self.exception("Failed to save to {}".format(file))
            raise IOError("Saving file failed")