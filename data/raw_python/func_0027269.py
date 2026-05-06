def _apply_callback(cls, callback, result):
        """ Synchronously execute callback """
        if not callback.immutable:
            callback.args = (result.id, ) + callback.args
        callback.apply()