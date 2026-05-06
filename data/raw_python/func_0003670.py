def init_board(self):
        """Init a valid board by given settings.

        Parameters
        ----------
        mine_map : numpy.ndarray
            the map that defines the mine
            0 is empty, 1 is mine
        info_map : numpy.ndarray
            the map that presents to gamer
            0-8 is number of mines in srrounding.
            9 is flagged field.
            10 is questioned field.
            11 is undiscovered field.
            12 is a mine field.
        """
        self.mine_map = np.zeros((self.board_height, self.board_width),
                                 dtype=np.uint8)
        idx_list = np.random.permutation(self.board_width*self.board_height)
        idx_list = idx_list[:self.num_mines]

        for idx in idx_list:
            idx_x = int(idx % self.board_width)
            idx_y = int(idx / self.board_width)

            self.mine_map[idx_y, idx_x] = 1

        self.info_map = np.ones((self.board_height, self.board_width),
                                dtype=np.uint8)*11