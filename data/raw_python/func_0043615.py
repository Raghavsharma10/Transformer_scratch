def set_read_only(self, value):
        """
        Sets whether model could be modified or not
        """
        if self.__read_only__ != value:
            self.__read_only__ = value
            self._update_read_only()