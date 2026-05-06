def create_rules(self):
        '''Adds rules for the command line options'''
        dmp = apps.get_app_config('django_mako_plus')
        # the default
        rules = [
            # files are included by default
            Rule('*',                                    level=None, filetype=TYPE_FILE,      score=1),
            # files at the app level are skipped
            Rule('*',                                    level=0,    filetype=TYPE_FILE,      score=-2),
            # directories are recursed by default
            Rule('*',                                    level=None, filetype=TYPE_DIRECTORY, score=1),
            # directories at the app level are skipped
            Rule('*',                                    level=0,    filetype=TYPE_DIRECTORY, score=-2),

            # media, scripts, styles directories are what we want to copy
            Rule('media',                                level=0,    filetype=TYPE_DIRECTORY, score=6),
            Rule('scripts',                              level=0,    filetype=TYPE_DIRECTORY, score=6),
            Rule('styles',                               level=0,    filetype=TYPE_DIRECTORY, score=6),

            # ignore the template cache directories
            Rule(dmp.options['TEMPLATES_CACHE_DIR'],     level=None, filetype=TYPE_DIRECTORY, score=-3),
            # ignore python cache directories
            Rule('__pycache__',                          level=None, filetype=TYPE_DIRECTORY, score=-3),
            # ignore compiled python files
            Rule('*.pyc',                                level=None, filetype=TYPE_FILE,      score=-3),
        ]
        # include rules have score of 50 because they trump all initial rules
        for pattern in (self.options.get('include_dir') or []):
            self.message('Setting rule - recurse directories: {}'.format(pattern), 1)
            rules.append(Rule(pattern, level=None, filetype=TYPE_DIRECTORY, score=50))
        for pattern in (self.options.get('include_file') or []):
            self.message('Setting rule - include files: {}'.format(pattern), 1)
            rules.append(Rule(pattern, level=None, filetype=TYPE_FILE, score=50))
        # skip rules have score of 100 because they trump everything, including the includes from the command line
        for pattern in (self.options.get('skip_dir') or []):
            self.message('Setting rule - skip directories: {}'.format(pattern), 1)
            rules.append(Rule(pattern, level=None, filetype=TYPE_DIRECTORY, score=-100))
        for pattern in (self.options.get('skip_file') or []):
            self.message('Setting rule - skip files: {}'.format(pattern), 1)
            rules.append(Rule(pattern, level=None, filetype=TYPE_FILE, score=-100))
        return rules