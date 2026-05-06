def search(self, keyword, source_in=None, **kwargs):
        """search song/artist/album by keyword

        TODO: search album or artist
        """
        for provider in self._providers:
            if source_in is not None:
                if provider.identifier not in source_in:
                    continue

            try:
                result = provider.search(keyword=keyword)
            except Exception as e:
                logger.exception(str(e))
                logger.error('Search %s in %s failed.' % (keyword, provider))
            else:
                yield result