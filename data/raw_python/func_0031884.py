def clear_silence(self, kwargs):
        """
        Clear a silence entry.
        """
        self._request('POST', '/silenced/clear', data=json.dumps(kwargs))
        return True