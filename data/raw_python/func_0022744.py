def register(self, toolkitname, *aliases):
        """Register a class to provide the event loop for a given GUI.
        
        This is intended to be used as a class decorator. It should be passed
        the names with which to register this GUI integration. The classes
        themselves should subclass :class:`InputHookBase`.
        
        ::
        
            @inputhook_manager.register('qt')
            class QtInputHook(InputHookBase):
                def enable(self, app=None):
                    ...
        """
        def decorator(cls):
            inst = cls(self)
            self.guihooks[toolkitname] = inst
            for a in aliases:
                self.aliases[a] = toolkitname
            return cls
        return decorator