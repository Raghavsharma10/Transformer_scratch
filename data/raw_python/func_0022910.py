def disconnect(self, callback=None):
        """ Disconnect the callback from this group. See
        :func:`connect() <vispy.event.EmitterGroup.connect>` and
        :func:`EventEmitter.connect() <vispy.event.EventEmitter.connect>` for
        more information.
        """
        ret = EventEmitter.disconnect(self, callback)
        if len(self._callbacks) == 0:
            self._connect_emitters(False)
        return ret