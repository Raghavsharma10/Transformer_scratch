def _load_json_file(self, file, decoder=None):
        """
        Load data from json file

        :param file: Readable file or path to file
        :type file: FileIO | str | unicode
        :param decoder: Use custom json decoder
        :type decoder: T <= flotils.loadable.DateTimeDecoder
        :return: Json data
        :rtype: None | int | float | str | list | dict
        :raises IOError: Failed to load
        """
        try:
            res = load_json_file(file, decoder=decoder)
        except ValueError as e:
            if "{}".format(e) == "No JSON object could be decoded":
                raise IOError("Decoding JSON failed")
            self.exception("Failed to load from {}".format(file))
            raise IOError("Loading file failed")
        except:
            self.exception("Failed to load from {}".format(file))
            raise IOError("Loading file failed")
        return res