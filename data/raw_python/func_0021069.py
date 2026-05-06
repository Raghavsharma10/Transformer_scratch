def insert(self, index, value):
        """
        Insert *value* into the collection at *index*.
        """
        if index == 0:
            return self._insert_left(value)

        def insert_middle_trans(pipe):
            self._insert_middle(index, value, pipe=pipe)

        return self._transaction(insert_middle_trans)