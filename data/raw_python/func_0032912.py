def _emit(self, event):
        """
        If the given event is a stat event, send a I{StatUpdate} command.
        """
        if (event.get('interface') is not iaxiom.IStatEvent and
            'athena_send_messages' not in event and
            'athena_received_messages' not in event):
            return

        out = []
        for k, v in event.iteritems():
            if k in ('system', 'message', 'interface', 'isError'):
                continue
            if not isinstance(v, unicode):
                v = str(v).decode('ascii')
            out.append(dict(key=k.decode('ascii'), value=v))
        self.callRemote(StatUpdate, data=out)