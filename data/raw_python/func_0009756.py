def cost(self, i, j):
        """Returns the distance between the end of path i
        and the start of path j."""
        return dist(self.endpoints[i][1], self.endpoints[j][0])