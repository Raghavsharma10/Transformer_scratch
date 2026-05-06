def load(filename=None,
        url=r"https://raw.githubusercontent.com/googlei18n/emoji4unicode/master/data/emoji4unicode.xml",
        loader_class=None):
    u"""load google's `emoji4unicode` project's xml file. must call this method first to use `e4u` library. this method never work twice if you want to reload, use `e4u.reload()` insted."""
    if not has_loaded():
        reload(filename, url, loader_class)