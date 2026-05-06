def zDDEInit(self):
        """Initiates link with OpticStudio DDE server"""
        self.pyver = _get_python_version()
        # do this only one time or when there is no channel
        if _PyZDDE.liveCh==0:
            try:
                _PyZDDE.server = _dde.CreateServer()
                _PyZDDE.server.Create("ZCLIENT")   
            except Exception as err:
                _sys.stderr.write("{}: DDE server may be in use!".format(str(err)))
                return -1
        # Try to create individual conversations for each ZEMAX application.
        self.conversation = _dde.CreateConversation(_PyZDDE.server)
        try:
            self.conversation.ConnectTo(self.appName, " ")
        except Exception as err:
            _sys.stderr.write("{}.\nOpticStudio UI may not be running!\n".format(str(err)))
            # should close the DDE server if it exist
            self.zDDEClose()
            return -1
        else:
            _PyZDDE.liveCh += 1 
            self.connection = True
            return 0