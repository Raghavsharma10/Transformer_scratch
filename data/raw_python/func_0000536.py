def on_doctree_read(app, document):
    """
    Hooks into Sphinx's ``doctree-read`` event.
    """
    literal_blocks = uqbar.book.sphinx.collect_literal_blocks(document)
    cache_mapping = uqbar.book.sphinx.group_literal_blocks_by_cache_path(literal_blocks)
    node_mapping = {}
    use_cache = bool(app.config["uqbar_book_use_cache"])
    for cache_path, literal_block_groups in cache_mapping.items():
        kwargs = dict(
            extensions=app.uqbar_book_extensions,
            setup_lines=app.config["uqbar_book_console_setup"],
            teardown_lines=app.config["uqbar_book_console_teardown"],
            use_black=bool(app.config["uqbar_book_use_black"]),
        )
        for literal_blocks in literal_block_groups:
            try:
                if use_cache:
                    local_node_mapping = uqbar.book.sphinx.interpret_code_blocks_with_cache(
                        literal_blocks, cache_path, app.connection, **kwargs
                    )
                else:
                    local_node_mapping = uqbar.book.sphinx.interpret_code_blocks(
                        literal_blocks, **kwargs
                    )
                node_mapping.update(local_node_mapping)
            except ConsoleError as exception:
                message = exception.args[0].splitlines()[-1]
                logger.warning(message, location=exception.args[1])
                if app.config["uqbar_book_strict"]:
                    raise
    uqbar.book.sphinx.rebuild_document(document, node_mapping)