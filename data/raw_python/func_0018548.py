def symlink_check_and_set(self):
        """
        The default symlink was changed from OMERO-CURRENT to OMERO.server.
        If `--sym` was not specified and OMERO-CURRENT exists in the current
        directory stop and warn.
        """
        if self.args.sym == '':
            if os.path.exists('OMERO-CURRENT'):
                log.error('Deprecated OMERO-CURRENT found but --sym not set')
                raise Stop(
                    30, 'The default for --sym has changed to OMERO.server '
                    'but the current directory contains OMERO-CURRENT. '
                    'Either remove OMERO-CURRENT or explicity pass --sym.')
        if self.args.sym in ('', 'auto'):
            self.args.sym = 'OMERO.server'