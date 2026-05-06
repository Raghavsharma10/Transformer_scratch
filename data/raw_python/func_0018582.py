def parse_mecab_options(self, options):
        '''Parses the MeCab options, returning them in a dictionary.

        Lattice-level option has been deprecated; please use marginal or nbest
        instead.

        :options string or dictionary of options to use when instantiating
                the MeCab instance. May be in short- or long-form, or in a
                Python dictionary.

        Returns:
            A dictionary of the specified MeCab options, where the keys are
            snake-cased names of the long-form of the option names.

        Raises:
            MeCabError: An invalid value for N-best was passed in.
        '''
        class MeCabArgumentParser(argparse.ArgumentParser):
            '''MeCab option parser for natto-py.'''

            def error(self, message):
                '''error(message: string)

                Raises ValueError.
                '''
                raise ValueError(message)

        options = options or {}
        dopts = {}

        if type(options) is dict:
            for name in iter(list(self._SUPPORTED_OPTS.values())):
                if name in options:
                    if options[name] or options[name] is '':
                        val = options[name]
                        if isinstance(val, bytes):
                            val = self.__bytes2str(options[name])
                        dopts[name] = val
        else:
            p = MeCabArgumentParser()
            p.add_argument('-r', '--rcfile',
                           help='use FILE as a resource file',
                           action='store', dest='rcfile')
            p.add_argument('-d', '--dicdir',
                           help='set DIR as a system dicdir',
                           action='store', dest='dicdir')
            p.add_argument('-u', '--userdic',
                           help='use FILE as a user dictionary',
                           action='store', dest='userdic')
            p.add_argument('-l', '--lattice-level',
                           help='lattice information level (DEPRECATED)',
                           action='store', dest='lattice_level', type=int)
            p.add_argument('-O', '--output-format-type',
                           help='set output format type (wakati, none,...)',
                           action='store', dest='output_format_type')
            p.add_argument('-a', '--all-morphs',
                           help='output all morphs (default false)',
                           action='store_true', default=False)
            p.add_argument('-N', '--nbest',
                           help='output N best results (default 1)',
                           action='store', dest='nbest', type=int)
            p.add_argument('-p', '--partial',
                           help='partial parsing mode (default false)',
                           action='store_true', default=False)
            p.add_argument('-m', '--marginal',
                           help='output marginal probability (default false)',
                           action='store_true', default=False)
            p.add_argument('-M', '--max-grouping-size',
                           help=('maximum grouping size for unknown words '
                                 '(default 24)'),
                           action='store', dest='max_grouping_size', type=int)
            p.add_argument('-F', '--node-format',
                           help='use STR as the user-defined node format',
                           action='store', dest='node_format')
            p.add_argument('-U', '--unk-format',
                           help=('use STR as the user-defined unknown '
                                 'node format'),
                           action='store', dest='unk_format')
            p.add_argument('-B', '--bos-format',
                           help=('use STR as the user-defined '
                                 'beginning-of-sentence format'),
                           action='store', dest='bos_format')
            p.add_argument('-E', '--eos-format',
                           help=('use STR as the user-defined '
                                 'end-of-sentence format'),
                           action='store', dest='eos_format')
            p.add_argument('-S', '--eon-format',
                           help=('use STR as the user-defined end-of-NBest '
                                 'format'),
                           action='store', dest='eon_format')
            p.add_argument('-x', '--unk-feature',
                           help='use STR as the feature for unknown word',
                           action='store', dest='unk_feature')
            p.add_argument('-b', '--input-buffer-size',
                           help='set input buffer size (default 8192)',
                           action='store', dest='input_buffer_size', type=int)
            p.add_argument('-C', '--allocate-sentence',
                           help='allocate new memory for input sentence',
                           action='store_true', dest='allocate_sentence',
                           default=False)
            p.add_argument('-t', '--theta',
                           help=('set temperature parameter theta '
                                 '(default 0.75)'),
                           action='store', dest='theta', type=float)
            p.add_argument('-c', '--cost-factor',
                           help='set cost factor (default 700)',
                           action='store', dest='cost_factor', type=int)

            opts = p.parse_args([o.replace('\"', '').replace('\'', '') for o in options.split()])

            for name in iter(list(self._SUPPORTED_OPTS.values())):
                if hasattr(opts, name):
                    v = getattr(opts, name)
                    if v or v is '':
                        dopts[name] = v

        # final checks
        if 'nbest' in dopts \
            and (dopts['nbest'] < 1 or dopts['nbest'] > self._NBEST_MAX):
            logger.error(self._ERROR_NVALUE)
            raise ValueError(self._ERROR_NVALUE)

        # warning for lattice-level deprecation
        if 'lattice_level' in dopts:
            logger.warn('WARNING: {}\n'.format(self._WARN_LATTICE_LEVEL))

        return dopts