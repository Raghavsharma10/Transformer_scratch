def handle_endtag(self, tag):
        """
        Handler of ending tag processing (overrided, private)
        """
        self.log.debug( u'Encountered an end tag : {0}'.format(tag) )
        if tag in self.sanitizelist:
            self.level -= 1
            return
        if tag in self.unclosedTags:
            return
        if self.isNotPurify or tag in self.whitelist_keys:
            self.data.append(u'</%s>' % tag)