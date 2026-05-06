def getbroker():
        '''
        return settings dictionnary
        '''
        if not Configuration.broker_initialized:
            Configuration._initconf()
            Configuration.broker_settings = Configuration.settings['broker']
            Configuration.broker_initialized = True
        return Configuration.broker_settings