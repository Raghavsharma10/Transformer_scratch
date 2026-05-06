def proper_kwargs(self, section, kwargs):
        """Returns kwargs updated with proper meta variables (like __assistant__).
        If this method is run repeatedly with the same section and the same kwargs,
        it always modifies kwargs in the same way.
        """
        kwargs['__section__'] = section
        kwargs['__assistant__'] = self
        kwargs['__env__'] = copy.deepcopy(os.environ)
        kwargs['__files__'] = [self._files]
        kwargs['__files_dir__'] = [self.files_dir]
        kwargs['__sourcefiles__'] = [self.path]
        # if any of the following fails, DA should keep running
        for i in ['system_name', 'system_version', 'distro_name', 'distro_version']:
            try:
                val = getattr(utils, 'get_' + i)()
            except:
                val = ''
            kwargs['__' + i + '__'] = val