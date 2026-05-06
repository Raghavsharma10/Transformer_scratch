def get_server_dir(self):
        """
        Either downloads and/or unzips the server if necessary
        return: the directory of the unzipped server
        """
        if not self.args.server:
            if self.args.skipunzip:
                raise Stop(0, 'Unzip disabled, exiting')

            log.info('Downloading server')

            # The downloader automatically symlinks the server, however if
            # we are upgrading we want to delay the symlink swap, so this
            # overrides args.sym
            # TODO: Find a nicer way to do this?
            artifact_args = copy.copy(self.args)
            artifact_args.sym = ''
            artifacts = Artifacts(artifact_args)
            server = artifacts.download('server')
        else:
            progress = 0
            if self.args.verbose:
                progress = 20
            ptype, server = fileutils.get_as_local_path(
                self.args.server, self.args.overwrite, progress=progress,
                httpuser=self.args.httpuser,
                httppassword=self.args.httppassword)
            if ptype == 'file':
                if self.args.skipunzip:
                    raise Stop(0, 'Unzip disabled, exiting')
                log.info('Unzipping %s', server)
                server = fileutils.unzip(
                    server, match_dir=True, destdir=self.args.unzipdir)

        log.debug('Server directory: %s', server)
        return server