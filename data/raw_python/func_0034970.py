def layout_circle(self):
    '''Position vertices evenly around a circle.'''
    n = self.num_vertices()
    t = np.linspace(0, 2*np.pi, n+1)[:n]
    return np.column_stack((np.cos(t), np.sin(t)))