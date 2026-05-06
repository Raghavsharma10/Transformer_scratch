def load(self, cfgFile=None, timeout=None):
        """expect that the data file has already been established"""
        #if cfgFile != None: self.cfgFile = cfgFile # if it's specified, use it
        if not cfgFile:
            cfgs = activeConfigs()
            if   len(cfgs) > 1: raise Exception("found too many configurations (%s); not clear which to load: %s"%(len(cfgs), cfgs))
            elif len(cfgs) < 1:
                if timeout: # wait for a configuration file to appear to be loaded
                    startWait = time.time()
                    timeReported = 0
                    while not cfgs:
                        timeWaited = time.time() - startWait
                        if timeWaited > timeout:
                            raise c.TimeoutExceeded("could not join game after %s seconds"%(timeout))
                        try:  cfgs = activeConfigs()
                        except:
                            if self.debug and timeWaited - timeReported >= 1:
                                timeReported += 1
                                print("second(s) waited for game to appear:  %d"%(timeReported))
                else:  raise Exception("must have a saved configuration to load or allow loading via timeout setting")
            cfgFile = cfgs.pop()
        try:
            with open(cfgFile, "rb") as f:
                data = f.read() # bytes => str
        except TypeError as e:
            print("ERROR %s: %s %s"%(e, cfgFile, type(cfgFile)))
            raise
        self.loadJson(data) # str => dict
        if self.debug:
            print("configuration loaded: %s"%(self.name))
            self.display()