def cfn_viz(template, parameters={}, outputs={}, out=sys.stdout):
    """Render dot output for cloudformation.template in json format.
    """
    known_sg, open_sg = _analyze_sg(template['Resources'])
    (graph, edges) = _extract_graph(template.get('Description', ''),
                                    template['Resources'], known_sg, open_sg)
    graph['edges'].extend(edges)
    _handle_terminals(template, graph, 'Parameters', 'source', parameters)
    _handle_terminals(template, graph, 'Outputs', 'sink', outputs)
    graph['subgraphs'].append(_handle_pseudo_params(graph['edges']))

    _render(graph, out=out)