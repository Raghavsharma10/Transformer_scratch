def phantomjs(self, *args, **kwargs):
        '''
        Execute PhantomJS by giving ``args`` as command line arguments.

        If test are run in verbose mode (``-v/--verbosity`` = 2), it output:
          - the title as header (with separators before and after)
          - modules and test names
          - assertions results (with ``django.utils.termcolors`` support)

        In case of error, a JsTestException is raised to give details about javascript errors.
        '''
        separator = '=' * LINE_SIZE
        title = kwargs['title'] if 'title' in kwargs else 'phantomjs output'
        nb_spaces = (LINE_SIZE - len(title)) // 2

        if VERBOSE:
            print('')
            print(separator)
            print(' ' * nb_spaces + title)
            print(separator)
            sys.stdout.flush()

        with NamedTemporaryFile(delete=True) as cookies_file:
            cmd = ('phantomjs', '--cookies-file=%s' % cookies_file.name) + args
            if self.timeout:
                cmd += (str(self.timeout),)
            parser = TapParser(debug=VERBOSITY > 2)
            output = self.execute(cmd)

            for item in parser.parse(output):
                if VERBOSE:
                    print(item.display())
                    sys.stdout.flush()

        if VERBOSE:
            print(separator)
            sys.stdout.flush()

        failures = parser.suites.get_all_failures()
        if failures:
            raise JsTestException('Failed javascript assertions', failures)
        if self.returncode > 0:
            raise JsTestException('PhantomJS return with non-zero exit code (%s)' % self.returncode)