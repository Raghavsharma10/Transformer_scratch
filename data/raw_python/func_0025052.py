def size_filter(labeled_grid, min_size):
        """
        Remove labeled objects that do not meet size threshold criteria.

        Args:
            labeled_grid: 2D output from label method.
            min_size: minimum size of object in pixels.

        Returns:
            labeled grid with smaller objects removed.
        """
        out_grid = np.zeros(labeled_grid.shape, dtype=int)
        slices = find_objects(labeled_grid)
        j = 1
        for i, s in enumerate(slices):
            box = labeled_grid[s]
            size = np.count_nonzero(box.ravel() == (i + 1))
            if size >= min_size and box.shape[0] > 1 and box.shape[1] > 1:
                out_grid[np.where(labeled_grid == i + 1)] = j
                j += 1
        return out_grid