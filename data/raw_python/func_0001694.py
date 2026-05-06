def get_value_from_cfg(cfg_file):
        ''' initial the configuration with file that you specify 
            Sample usage:            
                config = get_value_from_cfg()            
            return:
                return a dict        -->config[section][option]  such as config["twsm"]["dut_ip"]                
        '''    
    
        if not os.path.isfile(cfg_file):
            return
    
        cfg = {}   
        config = ConfigParser.RawConfigParser()
        
        try:
            config.read(cfg_file)
        except Exception as e:
    #         raise Exception("\n\tcommon exception 1.2: Not a well format configuration file. error: '%s'" %(e))
            return        
        for section in config.sections():
            cfg[section] = {}
            for option in config.options(section):
                cfg[section][option]=config.get(section,option)
        return cfg