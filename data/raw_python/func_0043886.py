def reverse_toctree(app, doctree, docname):
    """Reverse the order of entries in the root toctree if 'glob' is used."""
    if docname == "changes":
        for node in doctree.traverse():
            if node.tagname == "toctree" and node.get("glob"):
                node["entries"].reverse()
                break