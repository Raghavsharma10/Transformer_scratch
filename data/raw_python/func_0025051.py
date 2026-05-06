def label(self, input_grid):
        """
        Label input grid with hysteresis method.

        Args:
            input_grid: 2D array of values.

        Returns:
            Labeled output grid.
        """
        unset = 0
        high_labels, num_labels = label(input_grid > self.high_thresh)
        region_ranking = np.argsort(maximum(input_grid, high_labels, index=np.arange(1, num_labels + 1)))[::-1]
        output_grid = np.zeros(input_grid.shape, dtype=int)
        stack = []
        for rank in region_ranking:
            label_num = rank + 1
            label_i, label_j = np.where(high_labels == label_num)
            for i in range(label_i.size):
                if output_grid[label_i[i], label_j[i]] == unset:
                    stack.append((label_i[i], label_j[i]))
            while len(stack) > 0:
                index = stack.pop()
                output_grid[index] = label_num
                for i in range(index[0] - 1, index[0] + 2):
                    for j in range(index[1] - 1, index[1] + 2):
                        if 0 <= i < output_grid.shape[0] and 0 <= j < output_grid.shape[1]:
                            if (input_grid[i, j] > self.low_thresh) and (output_grid[i, j] == unset):
                                stack.append((i, j))
        return output_grid