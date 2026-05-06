def waitPuppetCatalogToBeApplied(self, key, sleepTime=5):
        """ Function waitPuppetCatalogToBeApplied
        Wait for puppet catalog to be applied

        @param key: The host name or ID
        @return RETURN: None
        """
        # Wait for puppet catalog to be applied
        loop_stop = False
        while not loop_stop:
            status = self[key].getStatus()
            if status == 'No Changes' or status == 'Active':
                self.__printProgression__(True,
                                          key + ' creation: provisioning OK')
                loop_stop = True
            elif status == 'Error':
                self.__printProgression__(False,
                                          key + ' creation: Error - '
                                          'Error during provisioning')
                loop_stop = True
                return False
            else:
                self.__printProgression__('In progress',
                                          key + ' creation: provisioning ({})'
                                          .format(status),
                                          eol='\r')
            time.sleep(sleepTime)