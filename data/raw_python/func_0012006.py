def gen_remote_driver(executor, capabilities):
        ''' Generate remote drivers with desired capabilities(self.__caps) and command_executor
        @param executor: command executor for selenium remote driver
        @param capabilities: A dictionary of capabilities to request when starting the browser session.
        @return: remote driver
        '''        
        # selenium requires browser's driver and PATH env. Firefox's driver is required for selenium3.0            
        firefox_profile = capabilities.pop("firefox_profile",None)            
        return webdriver.Remote(executor, desired_capabilities=capabilities, browser_profile = firefox_profile)