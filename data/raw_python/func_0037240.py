def close(self):
        """close conn"""
        if not self._s or not hasattr(self._s, "close"):
            return
        try:
            self._s.close()
        except BaseException:
            pass