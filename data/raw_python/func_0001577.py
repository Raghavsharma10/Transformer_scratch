def generate(self):
        """Return a generator that yields pieces of XML."""
        # atom demands either an author element in every entry or a global one
        if not self.author:
            if False in map(lambda e: bool(e.author), self.entries):
                self.author = ({'name': u'unbekannter Autor'},)

        if not self.updated:
            dates = sorted([entry.updated for entry in self.entries])
            self.updated = dates and dates[-1] or datetime.utcnow()

        yield u'<?xml version="1.0" encoding="utf-8"?>\n'
        yield u'<feed xmlns="http://www.w3.org/2005/Atom">\n'
        yield '  ' + _make_text_block('title', self.title, self.title_type)
        yield u'  <id>%s</id>\n' % escape(self.id)
        yield u'  <updated>%s</updated>\n' % format_iso8601(self.updated, self.timezone)
        if self.url:
            yield u'  <link href="%s" />\n' % escape(self.url, True)
        if self.feed_url:
            yield u'  <link href="%s" rel="self" />\n' % \
                escape(self.feed_url, True)
        for link in self.links:
            yield u'  <link %s/>\n' % ''.join('%s="%s" ' % \
                (k, escape(link[k], True)) for k in link)
        for author in self.author:
            yield u'  <author>\n'
            yield u'    <name>%s</name>\n' % escape(author['name'])
            if 'uri' in author:
                yield u'    <uri>%s</uri>\n' % escape(author['uri'])
            if 'email' in author:
                yield '    <email>%s</email>\n' % escape(author['email'])
            yield '  </author>\n'
        if self.subtitle:
            yield '  ' + _make_text_block('subtitle', self.subtitle,
                                          self.subtitle_type)
        if self.icon:
            yield u'  <icon>%s</icon>\n' % escape(self.icon)
        if self.logo:
            yield u'  <logo>%s</logo>\n' % escape(self.logo)
        if self.rights:
            yield '  ' + _make_text_block('rights', self.rights,
                                          self.rights_type)
        generator_name, generator_url, generator_version = self.generator
        if generator_name or generator_url or generator_version:
            tmp = [u'  <generator']
            if generator_url:
                tmp.append(u' uri="%s"' % escape(generator_url, True))
            if generator_version:
                tmp.append(u' version="%s"' % escape(generator_version, True))
            tmp.append(u'>%s</generator>\n' % escape(generator_name))
            yield u''.join(tmp)
        for entry in self.entries:
            for line in entry.generate():
                yield u'  ' + line
        yield u'</feed>\n'