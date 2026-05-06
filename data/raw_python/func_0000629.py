def __check_existing(self, row):
        """Check if row exists in table
        """
        if self.__update_keys is not None:
            key = tuple(row[key] for key in self.__update_keys)
            if key in self.__bloom:
                return True
            self.__bloom.add(key)
            return False
        return False