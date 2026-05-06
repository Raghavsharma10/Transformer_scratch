def main_module_name() -> str:
    """Returns main module and module name pair."""
    if not hasattr(main_module, '__file__'):
        # running from interactive shell
        return None

    main_filename = os.path.basename(main_module.__file__)
    module_name, ext = os.path.splitext(main_filename)
    return module_name