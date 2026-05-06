def __get_charset(self):
        '''Return the character encoding (charset) used internally by MeCab.

        Charset is that of the system dictionary used by MeCab. Will defer to
        the user-specified MECAB_CHARSET environment variable, if set.

        Defaults to shift-jis on Windows.
        Defaults to utf-8 on Mac OS.
        Defaults to euc-jp, as per MeCab documentation, when all else fails.

        Returns:
            Character encoding (charset) used by MeCab.
        '''
        cset = os.getenv(self.MECAB_CHARSET)
        if cset:
            logger.debug(self._DEBUG_CSET_DEFAULT.format(cset))
            return cset
        else:
            try:
                res = Popen(['mecab', '-D'], stdout=PIPE).communicate()
                lines = res[0].decode()
                if not lines.startswith('unrecognized'):
                    dicinfo = lines.split(os.linesep)
                    t = [t for t in dicinfo if t.startswith('charset')]
                    if len(t) > 0:
                        cset = t[0].split()[1].lower()
                        logger.debug(self._DEBUG_CSET_DEFAULT.format(cset))
                        return cset
                    else:
                        logger.error('{}\n'.format(self._ERROR_NODIC))
                        raise EnvironmentError(self._ERROR_NODIC)
                else:
                    logger.error('{}\n'.format(self._ERROR_NOCMD))
                    raise EnvironmentError(self._ERROR_NOCMD)
            except OSError:
                cset = 'euc-jp'
                if sys.platform == 'win32':
                    cset = 'shift-jis'
                elif sys.platform == 'darwin':
                    cset = 'utf8'
                logger.debug(self._DEBUG_CSET_DEFAULT.format(cset))
                return cset