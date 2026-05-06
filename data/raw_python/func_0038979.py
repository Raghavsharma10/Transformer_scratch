def get(self, item, not_found_value=None):
        "Method like dict.get() which can return specified value if key not found"

        if item in self.keys:
            return self.__data[item]
        else:
            return not_found_value