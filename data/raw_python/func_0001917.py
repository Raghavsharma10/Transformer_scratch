def load_file(self, path):
        """
        Load file

        :param path: Path to file
        :type path: str | unicode
        :return: Loaded settings
        :rtype: None | str | unicode | int | list | dict
        :raises IOError: If file not found or error accessing file
        """
        res = None

        if not path:
            IOError("No path specified to save")

        if not os.path.isfile(path):
            raise IOError("File not found {}".format(path))

        try:
            with io.open(path, "r", encoding="utf-8") as f:
                if path.endswith(".json"):
                    res = self._load_json_file(f)
                elif path.endswith(".yaml") or path.endswith(".yml"):
                    res = self._load_yaml_file(f)
        except IOError:
            raise
        except Exception as e:
            self.exception("Failed reading {}".format(path))
            raise IOError(e)
        return res