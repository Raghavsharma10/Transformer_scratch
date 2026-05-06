def get_paths(self):
        '''
        get list of module paths
        '''

        # guess site package dir of virtualenv (system dependent)
        venv_site_packages = '%s/lib/site-packages' % self.venv_dir

        if not os.path.isdir(venv_site_packages):
            venv_site_packages_glob = glob.glob('%s/lib/*/site-packages' % self.venv_dir)

            if len(venv_site_packages_glob):
                venv_site_packages = venv_site_packages_glob[0]

        return [
            self.venv_dir,
            venv_site_packages
        ]