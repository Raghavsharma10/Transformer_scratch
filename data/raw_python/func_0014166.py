def conf(self, key=None, defval=None):
        '''return YunpianConf if key=None, else return value in YunpianConf'''
        if key is None:
            return self._ypconf
        val = self._ypconf.conf(key)
        return defval if val is None else val