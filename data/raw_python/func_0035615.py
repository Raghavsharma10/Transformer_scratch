def plot(self, coordinates, directed=False, weighted=False, fig='current',
           ax=None, edge_style=None, vertex_style=None, title=None, cmap=None):
    '''Plot the graph using matplotlib in 2 or 3 dimensions.

    coordinates : (n,2) or (n,3) array of vertex coordinates
    directed : if True, edges have arrows indicating direction.
    weighted : if True, edges are colored by their weight.
    fig : a matplotlib Figure to use, or one of {'new','current'}. Defaults to
          'current', which will call gcf(). Only used when ax=None.
    ax : a matplotlib Axes to use. Defaults to gca()
    edge_style : string or dict of styles for edges. Defaults to 'k-'
    vertex_style : string or dict of styles for vertices. Defaults to 'ko'
    title : string to display as the plot title
    cmap : a matplotlib Colormap to use for edge weight coloring
    '''
    X = np.atleast_2d(coordinates)
    assert 0 < X.shape[1] <= 3, 'too many dimensions to plot'
    if X.shape[1] == 1:
      X = np.column_stack((np.arange(X.shape[0]), X))
    is_3d = (X.shape[1] == 3)
    if ax is None:
      ax = _get_axis(is_3d, fig)
    edge_kwargs = dict(colors='k', linestyles='-', linewidths=1, zorder=1)
    vertex_kwargs = dict(marker='o', c='k', s=20, edgecolor='none', zorder=2)
    if edge_style is not None:
      if not isinstance(edge_style, dict):
        edge_style = _parse_fmt(edge_style, color_key='colors')
      edge_kwargs.update(edge_style)
    if vertex_style is not None:
      if not isinstance(vertex_style, dict):
        vertex_style = _parse_fmt(vertex_style, color_key='c')
      vertex_kwargs.update(vertex_style)
    if weighted and self.is_weighted():
      edge_kwargs['array'] = self.edge_weights()
    if directed and self.is_directed():
      _directed_edges(self, X, ax, is_3d, edge_kwargs, cmap)
    else:
      _undirected_edges(self, X, ax, is_3d, edge_kwargs, cmap)
    ax.scatter(*X.T, **vertex_kwargs)
    ax.autoscale_view()
    if title:
      ax.set_title(title)
    return pyplot.show