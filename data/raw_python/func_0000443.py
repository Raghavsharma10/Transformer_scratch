def html_visit_inheritance_diagram(
    self: NodeVisitor, node: inheritance_diagram
) -> None:
    """
    Builds HTML output from an :py:class:`~uqbar.sphinx.inheritance.inheritance_diagram` node.
    """
    inheritance_graph = node["graph"]
    urls = build_urls(self, node)
    graphviz_graph = inheritance_graph.build_graph(urls)
    dot_code = format(graphviz_graph, "graphviz")
    # TODO: We can perform unflattening here
    aspect_ratio = inheritance_graph.aspect_ratio
    if aspect_ratio:
        aspect_ratio = math.ceil(math.sqrt(aspect_ratio[1] / aspect_ratio[0]))
    if aspect_ratio > 1:
        process = subprocess.Popen(
            ["unflatten", "-l", str(aspect_ratio), "-c", str(aspect_ratio), "-f"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(dot_code.encode())
        dot_code = stdout.decode()
    render_dot_html(self, node, dot_code, {}, "inheritance", "inheritance")
    raise SkipNode