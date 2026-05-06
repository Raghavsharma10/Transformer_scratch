def mapping(self, struct, key_depth=1000, tree_depth=1,
                update_callable=None):
        """ Generates random values for dict-like objects

            @struct: the dict-like structure you want to fill with random data
            @size: #int number of random values to include in each @tree_depth
            @tree_depth: #int dict tree dimensions size, i.e.
                1=|{key: value}|
                2=|{key: {key: value}, key2: {key2: value2}}|
            @update_callable: #callable method which updates data in your
                dict-like structure - e.g. :meth:builtins.dict.update

            -> random @struct
            ..
                from collections import UserDict
                from vital.debug import RandData

                class MyDict(UserDict):
                    pass

                rd = RandData(int)

                my_dict = MyDict()
                rd.dict(my_dict, 3, 1, my_dict.update)
                # -> {
                #   'SE0ZNy0F6O': 42078648993195761,
                #   'pbK': 70822820981335987,
                #   '0A5Aa7': 17503122029338459}
            ..
        """
        if not tree_depth:
            return self._map_type()
        _struct = struct()
        add_struct = _struct.update if not update_callable \
            else getattr(_struct, update_callable)
        for x in range(key_depth):
            add_struct({
                self.randstr: self.mapping(
                    struct, key_depth, tree_depth-1, update_callable)
            })
        return _struct