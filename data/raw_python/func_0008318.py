def act(self):
        """
        Carries out the action associated with Stop button
        """
        g = get_root(self).globals
        g.clog.debug('Stop pressed')

        # Stop exposure meter
        # do this first, so timer doesn't also try to enable idle mode
        g.info.timer.stop()

        def stop_in_background():
            try:
                self.stopping = True
                if execCommand(g, 'abort'):
                    self.stopped_ok = True
                else:
                    g.clog.warn('Failed to stop run')
                    self.stopped_ok = False
                self.stopping = False
            except Exception as err:
                g.clog.warn('Failed to stop run. Error = ' + str(err))
                self.stopping = False
                self.stopped_ok = False

        # stopping can take a while during which the GUI freezes so run in
        # background.
        t = threading.Thread(target=stop_in_background)
        t.daemon = True
        t.start()
        self.after(500, self.check)