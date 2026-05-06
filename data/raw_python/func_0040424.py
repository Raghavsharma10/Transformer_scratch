def getclient():
        '''
        return settings dictionnary
        '''
        if not Configuration.client_initialized:
            Configuration._initconf()
            Configuration.client_settings = Configuration.settings['client']
            Configuration.client_initialized = True
        return Configuration.client_settings