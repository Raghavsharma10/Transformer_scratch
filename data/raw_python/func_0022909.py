def connect(self, callback, ref=False, position='first',
                before=None, after=None):
        """ Connect the callback to the event group. The callback will receive
        events from *all* of the emitters in the group.

        See :func:`EventEmitter.connect() <vispy.event.EventEmitter.connect>`
        for arguments.
        """
        self._connect_emitters(True)
        return EventEmitter.connect(self, callback, ref, position,
                                    before, after)