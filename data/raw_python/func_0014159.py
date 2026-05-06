def _init(self, clnt):
        '''initialize api by YunpianClient'''
        assert clnt, "clnt is None"
        self._clnt = clnt
        self._apikey = clnt.apikey()
        self._version = clnt.conf(YP_VERSION, defval=VERSION_V2)
        self._charset = clnt.conf(HTTP_CHARSET, defval=CHARSET_UTF8)
        self._name = self.__class__.__module__.split('.')[-1]