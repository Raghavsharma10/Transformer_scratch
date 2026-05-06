def sequence(self, struct, size=1000, tree_depth=1, append_callable=None):
        """ Generates random values for sequence-like objects

            @struct: the sequence-like structure you want to fill with random
                data
            @size: #int number of random values to include in each @tree_depth
            @tree_depth: #int dict tree dimensions size, i.e.
                1=|(value1, value2)|
                2=|((value1, value2), (value1, value2))|
            @append_callable: #callable method which appends/adds data to your
                sequence-like structure - e.g. :meth:list.append

            -> random @struct
            ..
                from collections import UserList
                from vital.debug import RandData

                class MySequence(UserList):
                    pass

                rd = RandData(int)

                my_seq = MySequence()
                rd.sequence(my_seq, 3, 1, my_seq.append)
                # -> [88508293836062443, 49097807561770961, 55043550817099444]
            ..
        """
        if not tree_depth:
            return self._map_type()
        _struct = struct()
        add_struct = _struct.append if not append_callable \
            else getattr(_struct, append_callable)
        for x in range(size):
            add_struct(self.sequence(
                struct, size, tree_depth-1, append_callable))
        return _struct