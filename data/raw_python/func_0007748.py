def populate(self):
        """ Populate this list with all views that take no arguments.
        """
        from django.conf import settings
        from django.core import urlresolvers

        self.append(("", ""))
        urlconf = settings.ROOT_URLCONF
        resolver = urlresolvers.RegexURLResolver(r'^/', urlconf)
        # Collect base level views
        for key, value in resolver.reverse_dict.items():
            if isinstance(key, basestring):
                args = value[0][0][1]
                url = "/" + value[0][0][0]
                self.append((key, " ".join(key.split("_"))))
        # Collect namespaces (TODO: merge these two sections into one)
        for namespace, url in resolver.namespace_dict.items():
            for key, value in url[1].reverse_dict.items():
                if isinstance(key, basestring):
                    args = value[0][0][1]
                    full_key = '%s:%s' % (namespace, key)
                    self.append((full_key, "%s: %s" % (namespace, " ".join(key.split("_")))))
        self.sort()