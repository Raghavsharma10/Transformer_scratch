def randtuple(self):
        """ -> a #tuple of random #int """
        return tuple(
            self.randint
            for x in range(0, self.random.randint(3, 10)))