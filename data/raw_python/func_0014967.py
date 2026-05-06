def create_graph_html(js_template, css_template, html_template=None):
    """ Create HTML code block given the graph Javascript and CSS. """
    if html_template is None:
        html_template = read_lib('html', 'graph')

    # Create div ID for the graph and give it to the JS and CSS templates so
    # they can reference the graph.
    graph_id = 'graph-{0}'.format(_get_random_id())
    js = populate_template(js_template, graph_id=graph_id)
    css = populate_template(css_template, graph_id=graph_id)

    return populate_template(
        html_template,
        graph_id=graph_id,
        css=css,
        js=js
    )