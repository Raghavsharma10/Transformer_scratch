def setup(app) -> Dict[str, Any]:
    """
    Sets up Sphinx extension.
    """
    app.setup_extension("sphinx.ext.graphviz")
    app.add_node(
        inheritance_diagram,
        html=(html_visit_inheritance_diagram, None),
        latex=(latex_visit_inheritance_diagram, None),
        man=(skip, None),
        texinfo=(skip, None),
        text=(skip, None),
    )
    app.add_directive("inheritance-diagram", InheritanceDiagram)
    return {
        "version": uqbar.__version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }