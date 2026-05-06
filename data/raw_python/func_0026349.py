async def find_backwards(self, stream_name, predicate, predicate_label='predicate'):
        """Return first event matching predicate, or None if none exists.

        Note: 'backwards', both here and in Event Store, means 'towards the
        event emitted furthest in the past'.
        """
        logger = self._logger.getChild(predicate_label)
        logger.info('Fetching first matching event')
        uri = self._head_uri
        try:
            page = await self._fetcher.fetch(uri)
        except HttpNotFoundError as e:
            raise StreamNotFoundError() from e
        while True:
            evt = next(page.iter_events_matching(predicate), None)
            if evt is not None:
                return evt

            uri = page.get_link("next")
            if uri is None:
                logger.warning("No matching event found")
                return None

            page = await self._fetcher.fetch(uri)