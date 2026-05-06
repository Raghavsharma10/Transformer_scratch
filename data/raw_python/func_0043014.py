def right(self, num=None):
        """
        WITH SLICES BEING FLAT, WE NEED A SIMPLE WAY TO SLICE FROM THE RIGHT [-num:]
        """
        if num == None:
            return FlatList([_get_list(self)[-1]])
        if num <= 0:
            return Null

        return FlatList(_get_list(self)[-num:])