def _assemble_flowtable(self, values):
        """ generate a flowtable from a tuple of descriptors.
        """
        values = map(lambda x: [] if x is None else x, values)
        src = values[0] + values[1]
        dst = values[2] + values[3]

        thistable = dict()
        for s in src:
            thistable[s] = dst
        return thistable