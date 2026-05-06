def _updateEndpoints(self,*args,**kwargs):
        """
        Updates all endpoints except the one from which this slot was called.

        Note: this method is probably not complete threadsafe. Maybe a lock is needed when setter self.ignoreEvents
        """

        sender = self.sender()
        if not self.ignoreEvents:
            self.ignoreEvents = True

            for binding in self.bindings.values():
                if binding.instanceId == id(sender):
                    continue
                
                if args: 
                    binding.setter(*args,**kwargs)
                else:
                    binding.setter(self.bindings[id(sender)].getter())

            self.ignoreEvents = False