def build_options_str(self, options):
        '''Returns a string concatenation of the MeCab options.

        Args:
            options: dictionary of options to use when instantiating the MeCab
                instance.

        Returns:
            A string concatenation of the options used when instantiating the
            MeCab instance, in long-form.
        '''
        opts = []
        for name in iter(list(self._SUPPORTED_OPTS.values())):
            if name in options:
                key = name.replace('_', '-')
                if key in self._BOOLEAN_OPTIONS:
                    if options[name]:
                        opts.append('--{}'.format(key))
                else:
                    opts.append('--{}={}'.format(key, options[name]))

        return self.__str2bytes(' '.join(opts))