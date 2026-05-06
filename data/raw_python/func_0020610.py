def dispatch_loader(scraper, loader_name):
    """
    Decorator that enforces one time loading for scrapers. The one time loading is applied
    to partial loaders, e.g. only parse and load the home team roster once. This is not
    meant to be used directly.
    
    :param scraper: property name (string) containing an object of type :py:class:`scrapr.ReportLoader`
    :param loader_name: name of method that does the scraping/parsing
    :returns: function wrapper
    """
    l = '.'.join([scraper, loader_name])
    def wrapper(f):
        @wraps(f)
        def wrapped(self, *f_args, **f_kwargs):
            if not hasattr(self, '_loaded'):
                self._loaded = { }
                
            already_loaded = self._loaded.setdefault(l, False)
            if not already_loaded:
                attr = getattr(self, scraper)
                self._loaded[l] = getattr(attr, loader_name)() is not None
            return f(self, *f_args, **f_kwargs)
        return wrapped
    return wrapper