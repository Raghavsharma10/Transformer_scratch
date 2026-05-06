def get_path(self, i):
        """Returns the path corresponding to the node i."""
        index = (i - 1) // 2
        reverse = (i - 1) % 2
        path = self.paths[index]
        if reverse:
            return path.reversed()
        else:
            return path