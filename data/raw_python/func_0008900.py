def fill_n_todo(self):
        """
        Calculate and record the number of edge pixels left to do on each tile
        """
        left = self.left
        right = self.right
        top = self.top
        bottom = self.bottom
        for i in xrange(self.n_chunks):
            self.n_todo.ravel()[i] = np.sum([left.ravel()[i].n_todo,
                                            right.ravel()[i].n_todo,
                                            top.ravel()[i].n_todo,
                                            bottom.ravel()[i].n_todo])