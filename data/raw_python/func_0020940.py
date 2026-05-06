def parse_entry(self, entry):
        """An attempt to parse pieces of an entry out w/o xpath, by looping
        over the entry root's children and slotting them into the right places.
        This is going to be way messier than SpeedParserEntries, and maybe
        less cleanly usable, but it should be faster."""

        e = feedparser.FeedParserDict()
        tag_map = self.tag_map
        nslookup = self.nslookup

        for child in entry.getchildren():
            if isinstance(child, etree._Comment):
                continue
            ns, tag = clean_ns(child.tag)
            mapping = tag_map.get(tag, None)
            if mapping:
                getattr(self, 'parse_%s' % mapping)(child, e, nslookup.get(ns, ns))
            if not ns:
                continue
            fulltag = '%s:%s' % (nslookup.get(ns, ''), tag)
            mapping = tag_map.get(fulltag, None)
            if mapping:
                getattr(self, 'parse_%s' % mapping)(child, e, nslookup[ns])

        lacks_summary = 'summary' not in e or e['summary'] is None
        lacks_content = 'content' not in e or not bool(e.get('content', None))

        if not lacks_summary and lacks_content:
            e['content'] = [{'value': e.summary}]

        # feedparser sometimes copies the first content value into the
        # summary field when summary was completely missing;  we want
        # to do that as well, but avoid the case where summary was given as ''
        if lacks_summary and not lacks_content:
            e['summary'] = e['content'][0]['value']

        if e.get('summary', False) is None:
            e['summary'] = u''

        # support feed entries that have a guid but no link
        if 'guid' in e and 'link' not in e:
            e['link'] = full_href(e['guid'], self.baseurl)

        return e