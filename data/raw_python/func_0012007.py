def gen_local_driver(browser, capabilities):
        ''' Generate localhost drivers with desired capabilities(self.__caps)
        @param browser:  firefox or chrome
        @param capabilities:  A dictionary of capabilities to request when starting the browser session.
        @return:  localhost driver
        '''
        if browser == "firefox":
            fp = capabilities.pop("firefox_profile",None)
            return webdriver.Firefox(desired_capabilities =capabilities, firefox_profile=fp)
                   
        elif browser == "chrome":            
            return webdriver.Chrome(desired_capabilities=capabilities)
        
        else:
            raise TypeError("Unsupport browser {}".format(browser))