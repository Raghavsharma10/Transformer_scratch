def mouse_area(self, handler, group=0, ident=None):
        """Adds a new MouseProxy for the given group to the 
        EventListener.mouse_proxies dict if it is not in there yet, and returns
        the (new) MouseProxy. In listen() all entries in the current group of
        mouse_proxies are used."""
        key = ident or id(handler)
        if key not in self.mouse_proxies[group]:
            self.mouse_proxies[group][key] = MouseProxy(handler, ident)
        return self.mouse_proxies[group][key]