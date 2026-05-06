def on_builder_inited(app):
    """
    Hooks into Sphinx's ``builder-inited`` event.
    """
    app.cache_db_path = ":memory:"
    if app.config["uqbar_book_use_cache"]:
        logger.info(bold("[uqbar-book]"), nonl=True)
        logger.info(" initializing cache db")
        app.connection = uqbar.book.sphinx.create_cache_db(app.cache_db_path)