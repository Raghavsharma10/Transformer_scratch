def reload(filename=None,
        url=r"https://raw.githubusercontent.com/googlei18n/emoji4unicode/master/data/emoji4unicode.xml",
        loader_class=None):
    u"""reload google's `emoji4unicode` project's xml file. must call this method first to use `e4u` library."""
    if loader_class is None:
        loader_class = loader.Loader
    global _loader
    _loader = loader_class()
    _loader.load(filename, url)