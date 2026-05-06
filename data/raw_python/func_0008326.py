def update(self):
        """
        Updates @ 10Hz to give smooth running clock, checks
        run status @0.2Hz to reduce load on servers.
        """
        g = get_root(self).globals
        try:
            self.count += 1
            delta = int(round(time.time() - self.startTime))
            self.configure(text='{0:<d} s'.format(delta))

            if self.count % 50 == 0:
                if not isRunActive(g):

                    # try and write FITS table before enabling start button, otherwise
                    # a new start will clear table
                    try:
                        insertFITSHDU(g)
                    except Exception as err:
                        g.clog.warn('Could not add FITS Table to run')
                        g.clog.warn(str(err))

                    g.observe.start.enable()
                    g.observe.stop.disable()
                    g.setup.ngcReset.enable()
                    g.setup.powerOn.disable()
                    g.setup.powerOff.enable()
                    g.clog.info('Timer detected stopped run')

                    warn_cmd = '/usr/bin/ssh observer@192.168.1.1 spd-say "\'exposure finished\'"'
                    subprocess.check_output(warn_cmd, shell=True, stderr=subprocess.PIPE)

                    # enable idle mode now run has stopped
                    g.clog.info('Setting chips to idle')
                    idle = {'appdata': {'app': 'Idle'}}
                    try:
                        success = postJSON(g, idle)
                        if not success:
                            raise Exception('postJSON returned false')
                    except Exception as err:
                        g.clog.warn('Failed to enable idle mode')
                        g.clog.warn(str(err))

                    g.clog.info('Stopping offsets (if running')
                    try:
                        success = stopNodding(g)
                        if not success:
                            raise Exception('failed to stop dithering')
                    except Exception as err:
                        g.clog.warn('Failed to stop GTC offset script')
                        g.clog.warn(str(err))

                    self.stop()
                    return

        except Exception as err:
            if self.count % 100 == 0:
                g.clog.warn('Timer.update: error = ' + str(err))

        self.id = self.after(100, self.update)