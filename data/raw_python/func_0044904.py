def tween(self, t):
        """
        t is number between 0 and 1 to indicate how far the tween has progressed
        """
        if t is None:
            return None

        if self.method in self.method_to_tween:
            return self.method_to_tween[self.method](t)
        elif self.method in self.method_1param:
            return self.method_1param[self.method](t, self.param1)
        elif self.method in self.method_2param:
            return self.method_2param[self.method](t, self.param1, self.param2)
        else:
            raise Exception("Unsupported tween method {0}".format(self.method))