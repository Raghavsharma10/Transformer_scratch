def pop(self, index=-1):
        """
        Retrieve the value at *index*, remove it from the collection, and
        return it.
        """
        if index == 0:
            return self._pop_left()
        elif index == -1:
            return self._pop_right()
        else:
            return self._pop_middle(index)