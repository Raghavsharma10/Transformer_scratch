def is_locked(self):
        """
        Returns whether model is locked
        """
        if not self.__locked__:
            return False
        elif self.get_parent():
            return self.get_parent().is_locked()

        return True