def setup(app) -> Dict[str, Any]:
    """
    Sets up Sphinx extension.
    """
    app.connect("doctree-read", on_doctree_read)
    app.connect("builder-inited", on_builder_inited)
    app.add_css_file("uqbar.css")
    app.add_node(
        nodes.classifier, override=True, html=(visit_classifier, depart_classifier)
    )
    app.add_node(
        nodes.definition, override=True, html=(visit_definition, depart_definition)
    )
    app.add_node(nodes.term, override=True, html=(visit_term, depart_term))
    return {
        "version": uqbar.__version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }