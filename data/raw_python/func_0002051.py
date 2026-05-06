def delete(self):
        """ Delete this source """
        r = self._client.request('DELETE', self.url)
        logger.info("delete(): %s", r.status_code)