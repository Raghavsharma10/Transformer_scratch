def not_right(self, num):
        """
        WITH SLICES BEING FLAT, WE NEED A SIMPLE WAY TO SLICE FROM THE LEFT [:-num:]
        """
        if num == None:
            return FlatList([_get_list(self)[:-1:]])
        if num <= 0:
            return FlatList.EMPTY

        return FlatList(_get_list(self)[:-num:])