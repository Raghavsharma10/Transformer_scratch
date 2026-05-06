def watch_source(self, source_id):
        """ Add a souce to the watchlist. """
        source_id = int(source_id)
        r = yield from self._send_cmd(
                "WATCH S[%d] ON" % (source_id, ))
        self._watched_source.add(source_id)
        return r