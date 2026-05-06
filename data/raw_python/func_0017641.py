def urls(self):
        "Returns a list of (value, URL) tuples."
        # First, check the urls() method for each plugin.
        plugin_urls = []
        for plugin_name, plugin in \
                                self.model.model_databrowse().plugins.items():
            urls = plugin.urls(plugin_name, self)
            if urls is not None:
                #plugin_urls.append(urls)
                values = self.values()
                return zip(self.values(), urls)
        if self.field.rel:
            m = EasyModel(self.model.site, self.field.rel.to)
            if self.field.rel.to in self.model.model_list:
                lst = []
                for value in self.values():
                    if value is None:
                        continue
                    url = mark_safe('%s%s/%s/objects/%s/' %
                                            (self.model.site.root_url,
                                             m.model._meta.app_label,
                                             m.model._meta.model_name,
                                             iri_to_uri(value._get_pk_val())))
                    lst.append((smart_text(value), url))
            else:
                lst = [(value, None) for value in self.values()]
        elif self.field.choices:
            lst = []
            for value in self.values():
                url = mark_safe('%s%s/%s/fields/%s/%s/' %
                                        (self.model.site.root_url,
                                         self.model.model._meta.app_label,
                                         self.model.model._meta.model_name,
                                         self.field.name,
                                         iri_to_uri(self.raw_value)))
                lst.append((value, url))
        elif isinstance(self.field, models.URLField):
            val = self.values()[0]
            lst = [(val, iri_to_uri(val))]
        else:
            lst = [(self.values()[0], None)]
        return lst