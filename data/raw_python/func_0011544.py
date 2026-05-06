def validate(self, folder, cleanup=False, validate_folder=True):
        ''' validate is the entrypoint to all validation, for
            a folder, config, or url. If a URL is found, it is
            cloned and cleaned up.
           :param validate_folder: ensures the folder name (github repo)
                                   matches.
        '''
         
        # Obtain any repository URL provided
        if folder.startswith('http') or 'github' in folder:
            folder = clone(folder, tmpdir=self.tmpdir)

        # Load config.json if provided directly
        elif os.path.basename(folder) == 'config.json':
            config = os.path.dirname(folder)
            return self._validate_config(config, validate_folder)

        # Otherwise, validate folder and cleanup
        valid = self._validate_folder(folder)
        if cleanup is True:
            shutil.rmtree(folder)
        return valid