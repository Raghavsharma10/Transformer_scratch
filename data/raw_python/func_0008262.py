def act(self):
        """
        Carries out action associated with start button
        """
        g = get_root(self).globals
        # check binning against overscan
        msg = """
        HiperCAM has an o/scan of 50 pixels.
        Your binning does not fit into this
        region. Some columns will contain a
        mix of o/scan and data.

        Click OK if you wish to continue."""
        if g.ipars.oscan():
            xbin, ybin = g.ipars.wframe.xbin.value(), g.ipars.wframe.ybin.value()
            if xbin not in (1, 2, 5, 10) or ybin not in (1, 2, 5, 10):
                if not messagebox.askokcancel('Binning alert', msg):
                    return False

        # Check instrument pars are OK
        if not g.ipars.check():
            g.clog.warn('Invalid instrument parameters; save failed.')
            return False

        # create JSON to post
        data = createJSON(g)

        # POST
        try:
            success = postJSON(g, data)
            if not success:
                raise Exception('postJSON returned False')
        except Exception as err:
            g.clog.warn("Failed to post data to servers")
            g.clog.warn(str(err))
            return False

        # Is nod enabled? Should we start GTC offsetter?
        try:
            success = startNodding(g, data)
            if not success:
                raise Exception('Failed to start dither: response was false')
        except Exception as err:
            g.clog.warn("Failed to start GTC offsetter")
            g.clog.warn(str(err))
            return False

        # START
        try:
            success = execCommand(g, 'start')
            if not success:
                raise Exception("Start command failed: check server response")
        except Exception as err:
            g.clog.warn('Failed to start run')
            g.clog.warn(str(err))
            return False

        # Send first offset if nodding enabled.
        # Initial trigger is sent after first offset, otherwise we'll hang indefinitely
        try:
            success = forceNod(g, data)
            if not success:
                raise Exception('Failed to send intitial offset and trigger - exposure will be paused indefinitely')
        except Exception as err:
            g.clog.warn('Run is paused indefinitely')
            g.clog.warn('use "ngcbCmd seq start" to fix')
            g.clog.warn(str(err))

        # Run successfully started.
        # enable stop button, disable Start
        # also make inactive until RunType select box makes active again
        # start run timer
        # finally, clear table which stores TCS info during this run
        self.disable()
        self.run_type_set = False
        g.observe.stop.enable()
        g.info.timer.start()
        g.info.clear_tcs_table()
        return True