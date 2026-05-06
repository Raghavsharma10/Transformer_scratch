def import_bundles(self, dir, detach=False, force=False):
        """
        Import bundles from a directory

        :param dir:
        :return:
        """

        import yaml

        fs = fsopendir(dir)

        bundles = []

        for f in fs.walkfiles(wildcard='bundle.yaml'):

            self.logger.info('Visiting {}'.format(f))
            config = yaml.load(fs.getcontents(f))

            if not config:
                self.logger.error("Failed to get a valid bundle configuration from '{}'".format(f))

            bid = config['identity']['id']

            try:
                b = self.bundle(bid)

            except NotFoundError:
                b = None

            if not b:
                b = self.new_from_bundle_config(config)
                self.logger.info('{} Loading New'.format(b.identity.fqname))
            else:
                self.logger.info('{} Loading Existing'.format(b.identity.fqname))

            source_url = os.path.dirname(fs.getsyspath(f))
            b.set_file_system(source_url=source_url)
            self.logger.info('{} Loading from {}'.format(b.identity.fqname, source_url))
            b.sync_in()

            if detach:
                self.logger.info('{} Detaching'.format(b.identity.fqname))
                b.set_file_system(source_url=None)

            if force:
                self.logger.info('{} Sync out'.format(b.identity.fqname))
                # FIXME. It won't actually sync out until re-starting the bundle.
                # The source_file_system is probably cached
                b = self.bundle(bid)
                b.sync_out()

            bundles.append(b)
            b.close()

        return bundles