def _extract_author(self):
        """ Extract author(s) if not already done. """

        if bool(self.author):
            return

        for pattern in self.config.author:

            items = self.parsed_tree.xpath(pattern)

            if isinstance(items, basestring):
                # In case xpath returns only one element.
                items = [items]

            for item in items:

                if isinstance(item, basestring):
                    # '_ElementStringResult' object has no attribute 'text'
                    stripped_author = unicode(item).strip()

                else:
                    try:
                        stripped_author = item.text.strip()

                    except AttributeError:
                        # We got a <div>…
                        stripped_author = etree.tostring(item)

                if stripped_author:
                    self.author.add(stripped_author)
                    LOGGER.info(u'Author extracted: %s.', stripped_author,
                                extra={'siteconfig': self.config.host})