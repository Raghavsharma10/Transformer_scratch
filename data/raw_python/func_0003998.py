def cart_to_zmat(self, coordinates):
        """Convert cartesian coordinates to ZMatrix format

           Argument:
             coordinates  --  Cartesian coordinates (numpy array Nx3)

           The coordinates must match with the graph that was used to initialize
           the ZMatrixGenerator object.
        """
        N = len(self.graph.numbers)
        if coordinates.shape != (N, 3):
            raise ValueError("The shape of the coordinates must be (%i, 3)" % N)
        result = np.zeros(N, dtype=self.dtype)
        for i in range(N):
            ref0 = self.old_index[i]
            rel1 = -1
            rel2 = -1
            rel3 = -1
            distance = 0
            angle = 0
            dihed = 0
            if i > 0:
                ref1 = self._get_new_ref([ref0])
                distance = np.linalg.norm(coordinates[ref0]-coordinates[ref1])
                rel1 = i - self.new_index[ref1]
            if i > 1:
                ref2 = self._get_new_ref([ref0, ref1])
                angle, = ic.bend_angle(coordinates[[ref0, ref1, ref2]])
                rel2 = i - self.new_index[ref2]
            if i > 2:
                ref3 = self._get_new_ref([ref0, ref1, ref2])
                dihed, = ic.dihed_angle(coordinates[[ref0, ref1, ref2, ref3]])
                rel3 = i - self.new_index[ref3]
            result[i] = (self.graph.numbers[i], distance, rel1, angle, rel2, dihed, rel3)
        return result