def __attrs_str(self, tag, attrs):
        """
        Build string of attributes list for tag
        """
        enabled = self.whitelist.get(tag, ['*'])
        all_attrs = '*' in enabled
        items = []
        for attr in attrs:
            key = attr[0]
            value = attr[1] or ''
            if all_attrs or key in enabled:
                items.append( u'%s="%s"' % (key, value,) )
        return u' '.join(items)