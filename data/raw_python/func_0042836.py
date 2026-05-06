def __check_lock(self):
        """
        Cheking whether the inspector is wrapped or not
        (1) MODULE_FUNCTION: Check whether both obj/prop has __SINONLOCK__/LOCK or not
        (2) MODULE:          Check whether obj has __SINONLOCK__ or not
        (3) FUNCTION:        Check whether function(mock as a class) has LOCK or not
        Raise:
            lock_error: when inspector has been wrapped
        """
        if self.args_type == "MODULE_FUNCTION":
            if hasattr(getattr(self.obj, self.prop), "LOCK"):
                ErrorHandler.lock_error(self.prop)
        elif self.args_type == "MODULE":
            if hasattr(self.obj, "__SINONLOCK__"):
                ErrorHandler.lock_error(self.obj)
        elif self.args_type == "FUNCTION":
            if hasattr(getattr(CPSCOPE, self.obj.__name__), "LOCK"):
                ErrorHandler.lock_error(self.obj)