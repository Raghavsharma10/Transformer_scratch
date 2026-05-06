def hook_for(self, configurable_class, action):
        """
        Helper method for determining if an on_<configurable class>_<action>
        method is present, to be used as a hook in the add/update/remove
        configurable methods.
        """
        configurable_class_name = configurable_class.__name__.lower()

        return getattr(
            self,
            "on_" + configurable_class_name + "_" + action,
            None
        )