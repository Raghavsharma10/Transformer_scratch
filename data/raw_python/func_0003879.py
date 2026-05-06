def _consistent(self):
        """Checks the constency between self.__index and self.__order"""
        if len(self.__order) != sum(len(values) for values in self.__index.values()):
            return False
        import copy
        tmp = copy.copy(self.__order)
        for key, values in self.__index.items():
            for value in values:
                if value.name != key:
                    return False
                if value in tmp:
                    tmp.remove(value)
                else:
                    return False
                if isinstance(value, CP2KSection):
                    if not value._consistent():
                        return False
        return True