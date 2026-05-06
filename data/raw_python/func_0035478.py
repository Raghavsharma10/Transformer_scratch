def bandwidth(self):
    """Computes the 'bandwidth' of a graph."""
    return np.abs(np.diff(self.pairs(), axis=1)).max()