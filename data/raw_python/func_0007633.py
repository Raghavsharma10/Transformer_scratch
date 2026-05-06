def hoist_event(self, e):
        """ Hoist an xcb_generic_event_t to the right xcffib structure. """
        if e.response_type == 0:
            return self._process_error(ffi.cast("xcb_generic_error_t *", e))

        # We mask off the high bit here because events sent with SendEvent have
        # this bit set. We don't actually care where the event came from, so we
        # just throw this away. Maybe we could expose this, if anyone actually
        # cares about it.
        event = self._event_offsets[e.response_type & 0x7f]

        buf = CffiUnpacker(e)
        return event(buf)