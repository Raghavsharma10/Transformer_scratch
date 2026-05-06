def fix_shapes(self):
        """
        Fixes the shape of the data fields on edges. Left edges should be
        column vectors, and top edges should be row vectors, for example.
        """
        for i in xrange(self.n_chunks):
            for side in ['left', 'right', 'top', 'bottom']:
                edge = getattr(self, side).ravel()[i]
                if side in ['left', 'right']:
                    shp = [edge.todo.size, 1]
                else:
                    shp = [1, edge.todo.size]
                edge.done = edge.done.reshape(shp)
                edge.data = edge.data.reshape(shp)
                edge.todo = edge.todo.reshape(shp)