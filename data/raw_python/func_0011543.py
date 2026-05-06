def _validate_folder(self, folder=None):
        ''' validate folder takes a cloned github repo, ensures
            the existence of the config.json, and validates it.
        '''
        from expfactory.experiment import load_experiment

        if folder is None:
            folder=os.path.abspath(os.getcwd())

        config = load_experiment(folder, return_path=True)

        if not config:
            return notvalid("%s is not an experiment." %(folder))

        return self._validate_config(folder)