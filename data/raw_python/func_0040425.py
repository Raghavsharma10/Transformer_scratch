def getworker():
        '''
        return settings dictionnary
        '''
        if not Configuration.worker_initialized:
            Configuration._initconf()
            Configuration.worker_settings = Configuration.settings['worker']
            Configuration.worker_initialized = True
        return Configuration.worker_settings