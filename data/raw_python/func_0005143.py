def randset(self):
        """ -> a #set of random integers """
        return {
            self._map_type(int)
            for x in range(self.random.randint(3, 10))}