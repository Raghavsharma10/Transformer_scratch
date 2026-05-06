def stdin_read(self, data):
        """read Event (on channel ``stdin``)
        This is the event handler for ``read`` events specifically from the
        ``stdin`` channel. This is triggered each time stdin has data that
        it has read.
        """

        data = data.strip().decode("utf-8")
        self.log("Incoming:", data, lvl=verbose)

        if len(data) == 0:
            self.log('Use /help to get a list of enabled cli hooks')
            return

        if data[0] == "/":
            cmd = data[1:]
            args = []
            if ' ' in cmd:
                cmd, args = cmd.split(' ', maxsplit=1)
                args = args.split(' ')
            if cmd in self.hooks:
                self.log('Firing hooked event:', cmd, args, lvl=debug)
                self.fireEvent(self.hooks[cmd](*args))
            # TODO: Move these out, so we get a simple logic here
            elif cmd == 'frontend':
                self.log("Sending %s frontend rebuild event" %
                         ("(forced)" if 'force' in args else ''))
                self.fireEvent(
                    frontendbuildrequest(force='force' in args,
                                         install='install' in args),
                    "setup")
            elif cmd == 'backend':
                self.log("Sending backend reload event")
                self.fireEvent(componentupdaterequest(force=False), "setup")
            else:
                self.log('Unknown Command:', cmd, '. Use /help to get a list of enabled '
                                                  'cli hooks')