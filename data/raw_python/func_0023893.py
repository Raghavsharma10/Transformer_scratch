def debug(self, msg):
        '''
        Handle the debugging to a file
        '''
        # If debug is not disabled
        if self.__debug is not False:

            # If never was set, try to set it up
            if self.__debug is None:

                # Check what do we have inside settings
                debug_filename = getattr(settings, "AD_DEBUG_FILE", None)
                if debug_filename:
                    # Open the debug file pointer
                    self.__debug = open(settings.AD_DEBUG_FILE, 'a')
                else:
                    # Disable debuging forever
                    self.__debug = False

            if self.__debug:
                # Debug the given message
                self.__debug.write("{}\n".format(msg))
                self.__debug.flush()