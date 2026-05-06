def _parse_tensor(self, indices=False):
        '''Parse a tensor.'''
        if indices:
            self.line = self._skip_lines(1)

        tensor = np.zeros((3, 3))
        for i in range(3):
            tokens = self.line.split()
            if indices:
                tensor[i][0] = float(tokens[1])
                tensor[i][1] = float(tokens[2])
                tensor[i][2] = float(tokens[3])
            else:
                tensor[i][0] = float(tokens[0])
                tensor[i][1] = float(tokens[1])
                tensor[i][2] = float(tokens[2])
            self.line = self._skip_lines(1)
        return tensor