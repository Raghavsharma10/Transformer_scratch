def on_build_finished(app, exception):
    """
    Hooks into Sphinx's ``build-finished`` event.
    """
    if not app.config["uqbar_book_use_cache"]:
        return
    logger.info("")
    for row in app.connection.execute("SELECT path, hits FROM cache ORDER BY path"):
        path, hits = row
        if not hits:
            continue
        logger.info(bold("[uqbar-book]"), nonl=True)
        logger.info(" Cache hits for {}: {}".format(path, hits))