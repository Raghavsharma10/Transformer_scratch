def adjacent_tiles(self, tile, pattern):
            """This will return a list of the tiles adjacent to a given tile.
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Inputs:
                tile - This is the tile object for which the method will find
                    adjacent tiles.

                pattern - This will designate the pattern type that you want the
                    method to return

                    'p' = plus sign
                    'x' = diagonal
                    'b' = box

            (doc string updated ver 0.1)
            """

            # Initialize the list of tiles to return
            adj_tiles = []

            # Find the row and column of the input tile
            for i in self:
                for j in i:
                    if j == tile:
                        row = self.index(i)
                        column = self[row].index(j)

            # Define functions for the 2 distinct patterns
            def plus_sign(self, row, column):
                nonlocal adj_tiles
                if row - 1 >= 0:
                    adj_tiles += [self[row - 1][column]]
                if row + 1 != len(self):
                    adj_tiles += [self[row + 1][column]]
                if column - 1 >= 0:
                    adj_tiles += [self[row][column - 1]]
                if column + 1 != len(self[row]):
                    adj_tiles += [self[row][column + 1]]

            def diagonal(self, row, column):
                nonlocal adj_tiles
                if column - 1 >= 0:
                    if row - 1 >= 0:
                        adj_tiles += [self[row - 1][column - 1]]
                    if row + 1 != len(self):
                        adj_tiles += [self[row + 1][column - 1]]
                if column + 1 != len(self[row]):
                    if row - 1 >= 0:
                        adj_tiles += [self[row - 1][column + 1]]
                    if row + 1 != len(self):
                        adj_tiles += [self[row + 1][column + 1]]

            # Return the tiles that form a plus sign with the given input tile
            if pattern == 'p':
                plus_sign(self, row, column)

            # Return the tiles touching the four corners of the input tile
            elif pattern == 'x':
                diagonal(self, row, column)

            # Return all of the tiles surrounding the input tile
            elif pattern == 'b':
                plus_sign(self, row, column)
                diagonal(self, row, column)

            return adj_tiles