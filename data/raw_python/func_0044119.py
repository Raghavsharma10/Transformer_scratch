def load_modules(modules):
    """Load a module."""

    for dotted_module in modules:
        try:
            __import__(dotted_module)

        except ImportError as e:
            LOG.error("Unable to import %s: %s", dotted_module, e)