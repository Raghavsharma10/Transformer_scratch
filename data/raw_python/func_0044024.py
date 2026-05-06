def _extract_date(self):
        """ Extract date from HTML. """

        if self.date:
            return

        found = False

        for pattern in self.config.date:

            items = self.parsed_tree.xpath(pattern)

            if isinstance(items, basestring):
                # In case xpath returns only one element.
                items = [items]

            for item in items:
                if isinstance(item, basestring):
                    # '_ElementStringResult' object has no attribute 'text'
                    stripped_date = unicode(item).strip()

                else:
                    try:
                        stripped_date = item.text.strip()

                    except AttributeError:
                        # .text is None. We got a <div> item with span-only
                        # content. The result will probably be completely
                        # useless to a python developer, but at least we
                        # didn't fail handling the siteconfig directive.
                        stripped_date = etree.tostring(item)

                if stripped_date:
                    # self.date = strtotime(trim(elems, "; \t\n\r\0\x0B"))
                    self.date = stripped_date
                    LOGGER.info(u'Date extracted: %s.', stripped_date,
                                extra={'siteconfig': self.config.host})
                    found = True
                    break

            if found:
                break