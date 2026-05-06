def as_dict(self, section='Main', **kwargs):
        """Return template context from configs.

        """
        items = super(MakesiteParser, self).items(section, **kwargs)
        return dict(items)