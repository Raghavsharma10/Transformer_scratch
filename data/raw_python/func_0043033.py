def _maybe_purge_cache(self):
        """
        If enough time since last check has elapsed, check if any
        of the cached templates has changed. If any of the template
        files were deleted, remove that file only. If any were
        changed, then purge the entire cache.
        """

        if self._last_reload_check + MIN_CHECK_INTERVAL > time.time():
            return

        for name, tmpl in list(self.cache.items()):
            if not os.stat(tmpl.path):
                self.cache.pop(name)
                continue

            if os.stat(tmpl.path).st_mtime > tmpl.mtime:
                self.cache.clear()
                break

        self._last_reload_check = time.time()