def __get_libpath(self):
        '''Return the absolute path to the MeCab library.

        On Windows, the path to the system dictionary is used to deduce the
        path to libmecab.dll.

        Otherwise, mecab-config is used find the libmecab shared object or
        dynamic library (*NIX or Mac OS, respectively).

        Will defer to the user-specified MECAB_PATH environment variable, if
        set.

        Returns:
            The absolute path to the MeCab library.

        Raises:
            EnvironmentError: A problem was encountered in trying to locate the
                MeCab library.
        '''
        libp = os.getenv(self.MECAB_PATH)
        if libp:
            return os.path.abspath(libp)
        else:
            plat = sys.platform
            if plat == 'win32':
                lib = self._LIBMECAB.format(self._WINLIB_EXT)

                try:
                    v = self.__regkey_value(self._WINHKEY, self._WINVALUE)
                    ldir = v.split('etc')[0]
                    libp = os.path.join(ldir, 'bin', lib)
                except EnvironmentError as err:
                    logger.error('{}\n'.format(err))
                    logger.error('{}\n'.format(sys.exc_info()[0]))
                    raise EnvironmentError(
                        self._ERROR_WINREG.format(self._WINVALUE,
                                                  self._WINHKEY))
            else:
                # UNIX-y OS?
                if plat == 'darwin':
                    lib = self._LIBMECAB.format(self._MACLIB_EXT)
                else:
                    lib = self._LIBMECAB.format(self._UNIXLIB_EXT)

                try:
                    cmd = ['mecab-config', '--libs-only-L']
                    res = Popen(cmd, stdout=PIPE).communicate()
                    lines = res[0].decode()
                    if not lines.startswith('unrecognized'):
                        linfo = lines.strip()
                        libp = os.path.join(linfo, lib)
                    else:
                        raise EnvironmentError(
                            self._ERROR_MECABCONFIG.format(lib))
                except EnvironmentError as err:
                    logger.error('{}\n'.format(err))
                    logger.error('{}\n'.format(sys.exc_info()[0]))
                    raise EnvironmentError(self._ERROR_NOLIB.format(lib))

            if libp and os.path.exists(libp):
                libp = os.path.abspath(libp)
                os.environ[self.MECAB_PATH] = libp
                return libp
            else:
                raise EnvironmentError(self._ERROR_NOLIB.format(libp))