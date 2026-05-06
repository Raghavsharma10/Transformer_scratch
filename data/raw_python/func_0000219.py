def clean_cache():
    """
    Python won't realise that new module has appeared in the runtime
    We need to clean the cache of module finders. Hacking again
    :return:
    """
    import importlib
    try:  # Python ver < 3.3
        vermod = importlib.import_module("versioneer")
        globals()["versioneer"] = vermod
    except ImportError:
        importlib.invalidate_caches()