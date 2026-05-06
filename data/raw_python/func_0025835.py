def _getGuiSettings(self):
        """ Return a dict (ConfigObj) of all user settings found in rcFile. """
        # Put the settings into a ConfigObj dict (don't use a config-spec)
        rcFile = self._rcDir+os.sep+APP_NAME.lower()+'.cfg'
        if os.path.exists(rcFile):
            try:
                return configobj.ConfigObj(rcFile)
            except:
                raise RuntimeError('Error parsing: '+os.path.realpath(rcFile))

            # tho, for simple types, unrepr=True eliminates need for .cfgspc
            # also, if we turn unrepr on, we don't need cfgGetBool
        else:
            return {}