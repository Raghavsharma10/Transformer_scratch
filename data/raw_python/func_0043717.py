def unwatch_source(self, source_id):
        """ Remove a souce from the watchlist. """
        source_id = int(source_id)
        self._watched_sources.remove(source_id)
        return (yield from
                self._send_cmd("WATCH S[%d] OFF" % (
                    source_id, )))