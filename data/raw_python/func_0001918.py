def save_file(self, path, data, readable=False):
        """
        Save to file

        :param path: File path to save
        :type path: str | unicode
        :param data: To save
        :type data: None | str | unicode | int | list | dict
        :param readable: Format file to be human readable (default: False)
        :type readable: bool
        :rtype: None
        :raises IOError: If empty path or error writing file
        """
        if not path:
            IOError("No path specified to save")

        try:
            with io.open(path, "w", encoding="utf-8") as f:
                if path.endswith(".json"):
                    self._save_json_file(
                        f,
                        data,
                        pretty=readable,
                        compact=(not readable),
                        sort=True
                    )
                elif path.endswith(".yaml") or path.endswith(".yml"):
                    self._save_yaml_file(f, data)
        except IOError:
            raise
        except Exception as e:
            self.exception("Failed writing {}".format(path))
            raise IOError(e)