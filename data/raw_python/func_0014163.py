def custom_conf(self, conf):
        '''custom apikey and http parameters'''
        if conf:
            for (key, val) in conf.items():
                self.__conf[key] = val
        return self