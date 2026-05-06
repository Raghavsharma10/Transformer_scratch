def act(self):
        """
        Power on action
        """
        g = get_root(self).globals
        g.clog.debug('Power on pressed')

        if execCommand(g, 'online'):
            g.clog.info('ESO server online')
            g.cpars['eso_server_online'] = True

            if not isPoweredOn(g):
                success = execCommand(g, 'pon')
                if not success:
                    g.clog.warn('Unable to power on CLDC')
                    return False

            # change other buttons
            self.disable()
            g.observe.start.enable()
            g.observe.stop.disable()
            g.setup.powerOff.enable()

            success = execCommand(g, 'seqStart')
            if not success:
                g.clog.warn('Failed to start sequencer after Power On.')

            try:
                g.info.run.configure(text='{0:03d}'.format(getRunNumber(g)))
            except Exception as err:
                g.clog.warn('Failed to determine run number at start of run')
                g.clog.warn(str(err))
                g.info.run.configure(text='UNDEF')
            return True
        else:
            g.clog.warn('Failed to bring server online')
            return False