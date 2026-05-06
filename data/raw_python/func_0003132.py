def render_json(result, cfg, **kwargs):
    """
    Render to output a result in JSON format
    """
    result = json.loads(result.decode('utf-8'))
    head = result['head']
    if 'results' not in result:
        if 'boolean' in result:
            r = u'Result: {}'.format(result['boolean'])
        else:
            r = u'Unsupported result: \n' + unicode(result)
        return {'data': {'text/plain': r},
                'metadata': {}}

    vars = head['vars']
    nrow = len(result['results']['bindings'])
    if cfg.dis == 'table':
        j = json_iterator(vars, result['results']['bindings'], set(cfg.lan),
                          add_vtype=cfg.typ)
        n, data = html_table(j, limit=cfg.lmt, withtype=cfg.typ)
        data += div('Total: {}, Shown: {}', nrow, n, css="tinfo")
        data = {'text/html': div(data)}
    else:
        result = json.dumps(result,
                            ensure_ascii=False, indent=2, sort_keys=True)
        data = {'text/plain': unicode(result)}

    return {'data': data,
            'metadata': {}}