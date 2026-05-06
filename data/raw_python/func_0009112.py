def fetch_assets(self):
        """ download bootstrap assets to control host.
        If present on the control host they will be uploaded to the target host during bootstrapping.
        """
        # allow overwrites from the commandline
        packages = set(
            env.instance.config.get('bootstrap-packages', '').split())
        packages.update(['python27'])
        cmd = env.instance.config.get('bootstrap-local-download-cmd', 'wget -c -O "{0.local}" "{0.url}"')
        items = sorted(self.bootstrap_files.items())
        for filename, asset in items:
            if asset.url:
                if not exists(dirname(asset.local)):
                    os.makedirs(dirname(asset.local))
                local(cmd.format(asset))
            if filename == 'packagesite.txz':
                # add packages to download
                items.extend(self._fetch_packages(asset.local, packages))