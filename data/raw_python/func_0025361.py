def create_random_ind_grow(self, depth=0):
        "Random individual using grow method"
        lst = []
        self._depth = depth
        self._create_random_ind_grow(depth=depth, output=lst)
        return lst