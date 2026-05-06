def on_config_inited(app, config):
    """
    Hooks into Sphinx's ``config-inited`` event.
    """
    extension_paths = config["uqbar_book_extensions"] or [
        "uqbar.book.extensions.GraphExtension"
    ]
    app.uqbar_book_extensions = []
    for extension_path in extension_paths:
        module_name, _, class_name = extension_path.rpartition(".")
        module = importlib.import_module(module_name)
        extension_class = getattr(module, class_name)
        extension_class.setup_sphinx(app)
        app.uqbar_book_extensions.append(extension_class)