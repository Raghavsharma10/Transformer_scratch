def setup(app):
    """Initialize Sphinx extension."""
    app.setup_extension('nbsphinx')
    app.add_source_suffix('.nblink', 'linked_jupyter_notebook')
    app.add_source_parser(LinkedNotebookParser)
    app.add_config_value('nbsphinx_link_target_root', None, rebuild='env')

    return {'version': __version__, 'parallel_read_safe': True}