def render_graph(result, cfg, **kwargs):
    """
    Render to output a result that can be parsed as an RDF graph
    """
    # Mapping from MIME types to formats accepted by RDFlib
    rdflib_formats = {'text/rdf+n3': 'n3',
                      'text/turtle': 'turtle',
                      'application/x-turtle': 'turtle',
                      'text/turtle': 'turtle',
                      'application/rdf+xml': 'xml',
                      'text/rdf': 'xml',
                      'application/rdf+xml': 'xml'}


    try:
        got = kwargs.get('format', 'text/rdf+n3')
        fmt = rdflib_formats[got]
    except KeyError:
        raise KrnlException('Unsupported format for graph processing: {!s}', got)

    g = ConjunctiveGraph()
    g.load(StringInputSource(result), format=fmt)

    display = cfg.dis[0] if is_collection(cfg.dis) else cfg.dis
    if display in ('png', 'svg'):
        try:
            literal = len(cfg.dis) > 1 and cfg.dis[1].startswith('withlit')
            opt = {'lang': cfg.lan, 'literal': literal, 'graphviz': []}
            data, metadata = draw_graph(g, fmt=display, options=opt)
            return {'data': data,
                    'metadata': metadata}
        except Exception as e:
            raise KrnlException('Exception while drawing graph: {!r}', e)
    elif display == 'table':
        it = rdf_iterator(g, set(cfg.lan), add_vtype=cfg.typ)
        n, data = html_table(it, limit=cfg.lmt, withtype=cfg.typ)
        data += div('Shown: {}, Total rows: {}', n if cfg.lmt else 'all',
                    len(g), css="tinfo")
        data = {'text/html': div(data)}
    elif len(g) == 0:
        data = {'text/html': div(div('empty graph', css='krn-warn'))}
    else:
        data = {'text/plain': g.serialize(format='nt').decode('utf-8')}

    return {'data': data,
            'metadata': {}}