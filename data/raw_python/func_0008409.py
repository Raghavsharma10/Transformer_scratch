def push(self, kv):
        """ Adds a new item from the given (key, value)-tuple.
            If the key exists, pushes the updated item to the head of the dict.
        """
        if kv[0] in self: 
            self.__delitem__(kv[0])
        self.__setitem__(kv[0], kv[1])