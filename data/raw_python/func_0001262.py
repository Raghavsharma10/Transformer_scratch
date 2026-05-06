def add_version_func(self, show_version):
        ''' Enable --version and -V to show version information '''
        if callable(show_version):
            self.__show_version_func = show_version
        else:
            self.__show_version_func = lambda cli, args: print(show_version)
        self.parser.add_argument("-V", "--version", action="store_true")