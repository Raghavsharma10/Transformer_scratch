def set_split_extents_by_tile_shape(self):
        """
        Sets split shape :attr:`split_shape` and
        split extents (:attr:`split_begs` and :attr:`split_ends`)
        from value of :attr:`tile_shape`.
        """
        self.split_shape = ((self.array_shape - 1) // self.tile_shape) + 1
        self.split_begs = [[], ] * len(self.array_shape)
        self.split_ends = [[], ] * len(self.array_shape)
        for i in range(len(self.array_shape)):
            self.split_begs[i] = _np.arange(0, self.array_shape[i], self.tile_shape[i])
            self.split_ends[i] = _np.zeros_like(self.split_begs[i])
            self.split_ends[i][0:-1] = self.split_begs[i][1:]
            self.split_ends[i][-1] = self.array_shape[i]