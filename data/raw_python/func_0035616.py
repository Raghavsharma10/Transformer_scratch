def to_html(self, html_file, directed=False, weighted=False, vertex_ids=None,
              vertex_colors=None, vertex_labels=None, width=900, height=600,
              title=None, svg_border='1px solid black'):
    '''Write the graph as a d3 force-directed layout SVG to an HTML file.

    html_file : str|file-like, writeable destination for the output HTML.
    vertex_ids : unique IDs for each vertex, defaults to arange(num_vertices).
    vertex_colors : numeric color mapping for vertices, optional.
    vertex_labels : class labels for vertices, optional.
    title : str, written above the SVG as an h1, optional.
    svg_border : str, CSS for the 'border' attribute of the SVG element.
    '''
    if directed:
      raise NotImplementedError('Directed graphs are NYI for HTML output.')
    if (vertex_colors is not None) and (vertex_labels is not None):
      raise ValueError('Supply only one of vertex_colors, vertex_labels')

    # set up vertices
    if vertex_ids is None:
      vertex_ids = np.arange(self.num_vertices())
    elif len(vertex_ids) != self.num_vertices():
      raise ValueError('len(vertex_ids) != num vertices.')

    if vertex_labels is not None:
      vlabels, vcolors = np.unique(vertex_labels, return_inverse=True)
      if len(vcolors) != len(vertex_ids):
        raise ValueError('len(vertex_labels) != num vertices.')
    elif vertex_colors is not None:
      vcolors = np.array(vertex_colors, dtype=float, copy=False)
      if len(vcolors) != len(vertex_ids):
        raise ValueError('len(vertex_colors) != num vertices.')
      vcolors -= vcolors.min()
      vcolors /= vcolors.max()
    else:
      vcolors = []

    node_json = []
    for name, c in zip_longest(vertex_ids, vcolors):
      if c is not None:
        node_json.append('{"id": "%s", "color": %s}' % (name, c))
      else:
        node_json.append('{"id": "%s"}' % name)

    # set up edges
    pairs = self.pairs(directed=directed)
    if weighted:
      weights = self.edge_weights(directed=directed, copy=True).astype(float)
      weights -= weights.min()
      weights /= weights.max()
    else:
      weights = np.zeros(len(pairs)) + 0.5

    edge_json = []
    for (i,j), w in zip(pairs, weights):
      edge_json.append('{"source": "%s", "target": "%s", "weight": %f}' % (
          vertex_ids[i], vertex_ids[j], w))

    # emit self-contained HTML
    if not hasattr(html_file, 'write'):
      fh = open(html_file, 'w')
    else:
      fh = html_file
    print(u'<!DOCTYPE html><meta charset="utf-8"><style>', file=fh)
    print(u'svg { border: %s; }' % svg_border, file=fh)
    if weighted:
      print(u'.links line { stroke-width: 2px; }', file=fh)
    else:
      print(u'.links line { stroke: #000; stroke-width: 2px; }', file=fh)
    print(u'.nodes circle { stroke: #fff; stroke-width: 1px; }', file=fh)
    print(u'</style>', file=fh)
    if title:
      print(u'<h1>%s</h1>' % title, file=fh)
    print(u'<svg width="%d" height="%d"></svg>' % (width, height), file=fh)
    print(u'<script src="https://d3js.org/d3.v4.min.js"></script>', file=fh)
    print(u'<script>', LAYOUT_JS, sep=u'\n', file=fh)
    if vertex_colors is not None:
      print(u'var vcolor=d3.scaleSequential(d3.interpolateViridis);', file=fh)
    elif vertex_labels is not None:
      scale = 'd3.schemeCategory%d' % (10 if len(vlabels) <= 10 else 20)
      print(u'var vcolor = d3.scaleOrdinal(%s);' % scale, file=fh)
    else:
      print(u'function vcolor(){ return "#1776b6"; }', file=fh)
    print(u'var sim=layout_graph({"nodes": [%s], "links": [%s]});</script>' % (
        ',\n'.join(node_json), ',\n'.join(edge_json)), file=fh)
    fh.flush()