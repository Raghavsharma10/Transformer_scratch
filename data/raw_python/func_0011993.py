def write(self, fn=None, sorted=False, wait=0):
        """write the contents of this config to fn or its __filename__.
        """
        config = ConfigParser(interpolation=None)
        if sorted==True: keys.sort()
        for key in self.__dict__.get('ordered_keys') or self.keys():
            config[key] = {}
            ks = self[key].keys()
            if sorted==True: ks.sort()
            for k in ks:
                if type(self[key][k])==list and self.__join_list__ is not None:
                    config[key][k] = self.__join_list__.join([v for v in self[key][k] if v!=''])
                else:
                    config[key][k] = str(self[key][k])
        fn = fn or self.__dict__.get('__filename__')
        # use advisory locking on this file
        i = 0
        while os.path.exists(fn+'.LOCK') and i < wait:
            i += 1
            time.sleep(1)
        if os.path.exists(fn+'.LOCK'):
            raise FileExistsError(fn + ' is locked for writing')
        else:
            with open(fn+'.LOCK', 'w') as lf:
                lf.write(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
            with open(fn, 'w') as f:
                config.write(f)
            os.remove(fn+'.LOCK')