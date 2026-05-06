def latex_visit_inheritance_diagram(
    self: NodeVisitor, node: inheritance_diagram
) -> None:
    """
    Builds LaTeX output from an :py:class:`~uqbar.sphinx.inheritance.inheritance_diagram` node.
    """
    inheritance_graph = node["graph"]
    graphviz_graph = inheritance_graph.build_graph()
    graphviz_graph.attributes["size"] = 6.0
    dot_code = format(graphviz_graph, "graphviz")
    render_dot_latex(self, node, dot_code, {}, "inheritance")
    raise SkipNode