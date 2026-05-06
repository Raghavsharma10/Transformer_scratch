def create_random_ind_full(self, depth=0):
        "Random individual using full method"
        lst = []
        self._create_random_ind_full(depth=depth, output=lst)
        return lst