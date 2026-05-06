def _load_cache(self):
        """
        the method is implemented for the purpose of optimization, byte positions will not be re-read from a file
        that has already been used, if the content of the file has changed, and the name has been left the same,
        the old version of byte offsets will be loaded
        :return: list of byte offsets from existing file
        """
        try:
            with open(self.__cache_path, 'rb') as f:
                return load(f)
        except FileNotFoundError:
            return
        except IsADirectoryError as e:
            raise IsADirectoryError(f'Please delete {self.__cache_path} directory') from e
        except (UnpicklingError, EOFError) as e:
            raise UnpicklingError(f'Invalid cache file {self.__cache_path}. Please delete it') from e