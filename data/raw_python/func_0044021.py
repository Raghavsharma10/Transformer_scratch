def _extract_next_page_link(self):
        """ Try to get next page link. """

        # HEADS UP: we do not abort if next_page_link is already set:
        #           we try to find next (eg. find 3 if already at page 2).

        for pattern in self.config.next_page_link:
            items = self.parsed_tree.xpath(pattern)

            if not items:
                continue

            if len(items) == 1:
                item = items[0]

                if 'href' in item.keys():
                    self.next_page_link = item.get('href')

                else:
                    self.next_page_link = item.text.strip()

                LOGGER.info(u'Found next page link: %s.',
                            self.next_page_link)

                # First found link is the good one.
                break

            else:
                LOGGER.warning(u'%s items for next-page link %s',
                               items, pattern,
                               extra={'siteconfig': self.config.host})