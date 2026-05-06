def randdeque(self):
        """ -> a :class:collections.deque of random #int """
        return deque(
            self.randint
            for x in range(0, self.random.randint(3, 10)))