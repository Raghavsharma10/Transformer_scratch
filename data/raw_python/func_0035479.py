def profile(self):
    """Measure of bandedness, also known as 'envelope size'."""
    leftmost_idx = np.argmax(self.matrix('dense').astype(bool), axis=0)
    return (np.arange(self.num_vertices()) - leftmost_idx).sum()