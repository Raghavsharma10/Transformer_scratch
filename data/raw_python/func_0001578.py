def generate(self):
        """Yields pieces of ATOM XML."""
        base = ''
        if self.xml_base:
            base = ' xml:base="%s"' % escape(self.xml_base, True)
        yield u'<entry%s>\n' % base
        yield u'  ' + _make_text_block('title', self.title, self.title_type)
        yield u'  <id>%s</id>\n' % escape(self.id)
        yield u'  <updated>%s</updated>\n' % format_iso8601(self.updated, self.timezone)
        if self.published:
            yield u'  <published>%s</published>\n' % \
                  format_iso8601(self.published, self.timezone)
        if self.url:
            yield u'  <link href="%s" />\n' % escape(self.url)
        for author in self.author:
            yield u'  <author>\n'
            yield u'    <name>%s</name>\n' % escape(author['name'])
            if 'uri' in author:
                yield u'    <uri>%s</uri>\n' % escape(author['uri'])
            if 'email' in author:
                yield u'    <email>%s</email>\n' % escape(author['email'])
            yield u'  </author>\n'
        for link in self.links:
            yield u'  <link %s/>\n' % ''.join('%s="%s" ' % \
                (k, escape(link[k], True)) for k in link)
        if self.summary:
            yield u'  ' + _make_text_block('summary', self.summary,
                                           self.summary_type)
        if self.content:
            if issubclass(self.content.__class__, dict):
                if "content" in self.content:
                    yield u'  <content %s>%s</content>\n' % (' '.join('%s="%s"' % \
                        (k, escape(self.content[k], True)) for k in self.content if k != "content"), escape(self.content["content"]))
                else:
                    yield u'  <content %s/>\n' % ' '.join('%s="%s" ' % \
                        (k, escape(self.content[k], True)) for k in self.content)
            else:
                yield u'  ' + _make_text_block('content', self.content,
                                           self.content_type)
        yield u'</entry>\n'