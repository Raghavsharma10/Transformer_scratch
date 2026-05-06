def _set_slots_to_null(self, cls):
        """
        WHY ARE SLOTS NOT ACCESIBLE UNTIL WE ASSIGN TO THEM?
        """
        if hasattr(cls, "__slots__"):
            for s in cls.__slots__:
                self.__setattr__(s, Null)
        for b in cls.__bases__:
            self._set_slots_to_null(b)