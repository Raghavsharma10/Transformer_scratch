def __set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        """
        it works with seed labels:
        0: nothing
        1: object 1 - full seeds
        2: object 2 - full seeds
        3: object 1 - not a training seeds
        4: object 2 - not a training seeds
        """
        seeds_mask = (seeds == 1) | (seeds == 3)
        tdata2[seeds_mask] = np.max(tdata2) + 1
        tdata1[seeds_mask] = 0

        seeds_mask = (seeds == 2) | (seeds == 4)
        tdata1[seeds_mask] = np.max(tdata1) + 1
        tdata2[seeds_mask] = 0

        return tdata1, tdata2