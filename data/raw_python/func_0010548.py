def import_module(name: str) -> ModuleType:
    """Import module by it's name from following places in order:
      - main module
      - current working directory
      - Python path

    """
    logger.debug("Importing module: %s", name)
    if name == main_module_name():
        return main_module

    return importlib.import_module(name)