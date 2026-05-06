def get_merged_config(self):
        """Get merged config file.
        
        Returns an open StringIO containing the
        merged config file.
        """
        if self.yamldocs:
            return

        loadfiles = []
        if self.configfile:
            loadfiles.append(self.configfile)

        if self.configdir:
            # Gets list of all non-dotfile files from configdir.
            loadfiles.extend(
                [f for f in
                 [os.path.join(self.configdir, x) for x in
                  os.listdir(self.configdir)]
                 if os.path.isfile(f) and
                 not os.path.basename(f).startswith('.')])

        merged_configfile = io.StringIO()
        merged_configfile.write('-\n')
        for thefile in loadfiles:
            self.logdebug('reading in config file %s\n' % thefile)
            merged_configfile.write(open(thefile).read())
            merged_configfile.write('\n-\n')
        merged_configfile.seek(0)
        self.logdebug('merged log file: """\n%s\n"""\n' %
                      merged_configfile.read())
        merged_configfile.seek(0)
        return merged_configfile