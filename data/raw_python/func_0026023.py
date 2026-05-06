def setCmdline(self,value=1):
        """Set cmdline flag"""
        # set through dictionary to avoid extra calls to __setattr__
        if value:
            self.__dict__['flags'] = self.flags | _cmdlineFlag
        else:
            self.__dict__['flags'] = self.flags & ~_cmdlineFlag