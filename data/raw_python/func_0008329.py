def update_tcs(self):
        """
        Periodically update TCS info.

        A long running process, so run in a thread and fill a queue
        """
        g = get_root(self).globals

        if not g.cpars['tcs_on']:
            self.after(20000, self.update_tcs)
            return

        if g.cpars['telins_name'] == 'WHT':
            tcsfunc = tcs.getWhtTcs
        elif g.cpars['telins_name'] == 'GTC':
            tcsfunc = tcs.getGtcTcs
        else:
            g.clog.debug('TCS error: could not recognise ' +
                         g.cpars['telins_name'])
            return

        def tcs_threaded_update():
            try:
                ra, dec, pa, focus = tcsfunc()
                self.tcs_data_queue.put((ra, dec, pa, focus))
            except Exception as err:
                t, v, tb = sys.exc_info()
                error = traceback.format_exception_only(t, v)[0].strip()
                tback = 'TCS Traceback (most recent call last):\n' + \
                        ''.join(traceback.format_tb(tb))
                g.FIFO.put(('TCS', error, tback))

        t = threading.Thread(target=tcs_threaded_update)
        t.start()
        self.after(20000, self.update_tcs)