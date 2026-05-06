def check(self):
        """
        Checks the status of the stop exposure command
        This is run in background and can take a few seconds
        """
        g = get_root(self).globals
        if self.stopped_ok:
            # Exposure stopped OK; modify buttons
            self.disable()

            # try and write FITS table before enabling start button, otherwise
            # a new start will clear table
            try:
                insertFITSHDU(g)
            except Exception as err:
                g.clog.warn('Could not add FITS Table to run')
                g.clog.warn(str(err))

            g.observe.start.enable()
            g.setup.powerOn.disable()
            g.setup.powerOff.enable()

            # Report that run has stopped
            g.clog.info('Run stopped')

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
                    raise Exception('Failed to stop dithering: response was false')
            except Exception as err:
                g.clog.warn('Failed to stop GTC offset script')
                g.clog.warn(str(err))

            return True

        elif self.stopping:
            # Exposure in process of stopping
            # Disable lots of buttons
            self.disable()
            g.observe.start.disable()
            g.setup.powerOn.disable()
            g.setup.powerOff.disable()

            # wait a second before trying again
            self.after(500, self.check)

        else:
            self.enable()
            g.observe.start.disable()
            g.setup.powerOn.disable()
            g.setup.powerOff.disable()
            # Start exposure meter
            g.info.timer.start()
            return False