def read_chunk(self, t):
        """
        Read out the next chunk of memory

        Values in fortran binary streams begin and end with the number of bytes
        :param t: Data type (same format as used by struct).
        :return: Numpy array if the variable is an array, otherwise a scalar.
        """
        size = self.data[self.pos : self.pos + 4].cast("i")[0]
        d = self.data[self.pos + 4 : self.pos + 4 + size]

        assert (
            self.data[self.pos + 4 + size : self.pos + 4 + size + 4].cast("i")[0]
            == size
        )
        self.pos = self.pos + 4 + size + 4

        res = np.array(d.cast(t))

        # Return as a scalar or a numpy array if it is an array
        if res.size == 1:
            return res[0]
        return res