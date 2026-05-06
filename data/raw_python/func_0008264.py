def act(self):
        """
        Carries out the action associated with the Save button
        """
        g = get_root(self).globals
        g.clog.info('\nSaving current application to disk')

        # check instrument parameters
        if not g.ipars.check():
            g.clog.warn('Invalid instrument parameters; save failed.')
            return False

        # check run parameters
        rok, msg = g.rpars.check()
        if not rok:
            g.clog.warn('Invalid run parameters; save failed.')
            g.clog.warn(msg)
            return False

        # Get data to save
        data = createJSON(g, full=False)

        # Save to disk
        if saveJSON(g, data):
            # modify buttons
            g.observe.load.enable()
            g.observe.unfreeze.disable()

            # unfreeze the instrument and run params
            g.ipars.unfreeze()
            g.rpars.unfreeze()
            return True
        else:
            return False