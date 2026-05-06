def cancel(self):
        """ Cancel a pending publish task """
        target_url = self._client.get_url('PUBLISH', 'DELETE', 'single', {'id': self.id})
        r = self._client.request('DELETE', target_url)
        logger.info("cancel(): %s", r.status_code)