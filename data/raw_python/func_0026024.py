def setChanged(self,value=1):
        """Set changed flag"""
        # set through dictionary to avoid another call to __setattr__
        if value:
            self.__dict__['flags'] = self.flags | _changedFlag
        else:
            self.__dict__['flags'] = self.flags & ~_changedFlag