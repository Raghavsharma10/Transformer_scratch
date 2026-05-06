def handle_starttag(self, tag, attrs):
        """
        Handler of starting tag processing (overrided, private)
        """
        self.log.debug( u'Encountered a start tag: {0} {1}'.format(tag, attrs) )
        if tag in self.sanitizelist:
            self.level += 1
            return
        if self.isNotPurify or tag in self.whitelist_keys:
            attrs = self.__attrs_str(tag, attrs)
            attrs = ' ' + attrs if attrs else ''
            tmpl = u'<%s%s />' if tag in self.unclosedTags and self.isStrictHtml else u'<%s%s>'
            self.data.append( tmpl % (tag, attrs,) )