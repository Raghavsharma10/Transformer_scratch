def solve(self):
        """ Solve all possible positions of pieces within the context.

        Depth-first, tree-traversal of the product space.
        """
        # Create a new, empty board.
        board = Board(self.length, self.height)

        # Iterate through all combinations of positions.
        permutations = Permutations(self.pieces, self.vector_size)
        for positions in permutations:

            # Reuse board but flush all pieces.
            board.reset()

            for level, (piece_uid, linear_position) in enumerate(positions):
                # Try to place the piece on the board.
                try:
                    board.add(piece_uid, linear_position)
                # If one of the piece can't be added, throw the whole set, skip
                # the rotten branch and proceed to the next.
                except (OccupiedPosition, VulnerablePosition, AttackablePiece):
                    permutations.skip_branch(level)
                    break

            else:
                # All pieces fits, save solution and proceeed to the next
                # permutation.
                self.result_counter += 1
                yield board