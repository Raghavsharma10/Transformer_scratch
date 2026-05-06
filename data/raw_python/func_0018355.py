def normnorm(self):
        """
        Return a vecor noraml to this one with a norm of one

        :return: V2
        """

        n = self.norm()
        return V2(-self.y / n, self.x / n)