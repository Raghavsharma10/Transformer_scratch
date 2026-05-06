def _callOnOrganizerPlugins(self, methodName, *args):
        """
        Call a method on all L{IOrganizerPlugin} powerups on C{self.store}, or
        emit a deprecation warning for each one which does not implement that
        method.
        """
        for observer in self.getOrganizerPlugins():
            method = getattr(observer, methodName, None)
            if method is not None:
                method(*args)
            else:
                warn(
                    "IOrganizerPlugin now has the %s method, %s "
                    "did not implement it" % (methodName, observer.__class__,),
                    category=PendingDeprecationWarning)