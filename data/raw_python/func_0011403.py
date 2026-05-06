def _pfp__set_watch(self, watch_fields, update_func, *func_call_info):
        """Subscribe to update events on each field in ``watch_fields``, using
        ``update_func`` to update self's value when ``watch_field``
        changes"""
        self._pfp__watch_fields = watch_fields

        for watch_field in watch_fields:
            watch_field._pfp__watch(self)
        self._pfp__update_func = update_func
        self._pfp__update_func_call_info = func_call_info