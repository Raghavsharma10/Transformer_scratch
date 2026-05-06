def conf(self, key):
        '''get config'''
        return self.__conf[key] if key in self.__conf else _YunpianConf.YP_CONF.get(key)