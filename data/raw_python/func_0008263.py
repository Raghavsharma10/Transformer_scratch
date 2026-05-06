def act(self):
        """
        Carries out the action associated with the Load button
        """
        g = get_root(self).globals
        fname = filedialog.askopenfilename(
            defaultextension='.json',
            filetypes=[('json files', '.json'), ('fits files', '.fits')],
            initialdir=g.cpars['app_directory'])
        if not fname:
            g.clog.warn('Aborted load from disk')
            return False

        # load json
        if fname.endswith('.json'):
            with open(fname) as ifname:
                json_string = ifname.read()
        else:
            json_string = jsonFromFits(fname)

        # load up the instrument settings
        g.ipars.loadJSON(json_string)

        # load up the run parameters
        g.rpars.loadJSON(json_string)

        return True