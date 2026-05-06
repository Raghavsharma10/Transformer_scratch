def tick_all(self, times):
        """
        Tick all the EWMAs for the given number of times
        """

        for i in range(times):
            for avg in (self.m1, self.m5, self.m15, self.day):
                avg.tick()