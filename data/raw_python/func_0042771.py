def parse_args(self, doc, argv):
        """Parse ba arguments

        :param args: sys.argv[1:]
        :return: arguments
        """
        # first a little sneak peak if we have a generator
        arguments = docopt(doc, argv=argv, help=False)
        if arguments.get('<generator>'):
            name = arguments['<generator>']
            generator = self._get_generator(name)

            if hasattr(generator, 'DOC'):
                # register help for generator!
                # this runs after the generator was loaded so we have to
                # prepend the cmd!
                def _banana_help(args, router):
                    print(doc)

                cmd.register(lambda args: args.get('--help'), name,
                             _banana_help)
                doc = generator.DOC
                arguments = docopt(doc, argv=argv, help=False)

            # register generator '--version' cmd
            version = 'not provided by %s generator' % name
            if hasattr(generator, '__version__'):
                version = generator.__version__

            def _banana_version(args, router):
                print(version)

            cmd.register(lambda args: args.get('--version'), name,
                         _banana_version)

            # register generator interactive mode (last cmd for this generator)
            def _banana_run(args, router):
                router.navigate('run', name)
                router.navigate('exit')
            cmd.register(lambda args: True, name, _banana_run)

        return arguments