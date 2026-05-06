def insert(self, index, value):
        """
        Insert *value* into the collection at *index*.
        If the insertion would the collection to grow beyond ``maxlen``,
        raise ``IndexError``.
        """
        def insert_trans(pipe):
            len_self = self.__len__(pipe)
            if (self.maxlen is not None) and (len_self >= self.maxlen):
                raise IndexError

            if index == 0:
                self._insert_left(value, pipe)
            else:
                self._insert_middle(index, value, pipe=pipe)

        self._transaction(insert_trans)