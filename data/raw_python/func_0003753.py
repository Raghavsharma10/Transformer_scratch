def update_coordinates(self, coordinates=None):
        """Update the coordinates (and derived quantities)

           Argument:
             coordinates  --  new Cartesian coordinates of the system
        """
        if coordinates is not None:
            self.coordinates = coordinates
        self.numc = len(self.coordinates)
        self.distances = np.zeros((self.numc, self.numc), float)
        self.deltas = np.zeros((self.numc, self.numc, 3), float)
        self.directions = np.zeros((self.numc, self.numc, 3), float)
        self.dirouters = np.zeros((self.numc, self.numc, 3, 3), float)
        for index1, coordinate1 in enumerate(self.coordinates):
            for index2, coordinate2 in enumerate(self.coordinates):
                delta = coordinate1 - coordinate2
                self.deltas[index1, index2] = delta
                distance = np.linalg.norm(delta)
                self.distances[index1, index2] = distance
                if index1 != index2:
                    tmp = delta/distance
                    self.directions[index1, index2] = tmp
                    self.dirouters[index1, index2] = np.outer(tmp, tmp)