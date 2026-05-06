def volume(self):
        """The volume of the unit cell

           The actual definition of the volume depends on the number of active
           directions:

           * num_active == 0  --  always -1
           * num_active == 1  --  length of the cell vector
           * num_active == 2  --  surface of the parallelogram
           * num_active == 3  --  volume of the parallelepiped
        """
        active = self.active_inactive[0]
        if len(active) == 0:
            return -1
        elif len(active) == 1:
            return np.linalg.norm(self.matrix[:, active[0]])
        elif len(active) == 2:
            return np.linalg.norm(np.cross(self.matrix[:, active[0]], self.matrix[:, active[1]]))
        elif len(active) == 3:
            return abs(np.linalg.det(self.matrix))