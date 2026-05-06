def execute(self, *args, **options):
        '''Placing this in execute because then subclass handle() don't have to call super'''
        if options['verbose']:
            options['verbosity'] = 3
        if options['quiet']:
            options['verbosity'] = 0
        self.verbosity = options.get('verbosity', 1)
        super().execute(*args, **options)