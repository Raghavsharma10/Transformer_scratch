def randdict(self):
        """ -> a #dict of |{random_string: random_int}| """
        return {
            self.randstr: self._map_type(int)
            for x in range(self.random.randint(3, 10))}