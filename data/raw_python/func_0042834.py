def _get_wrapper(self):
        """
        Return:
            Wrapper object
        Raise:
            Exception if wrapper object cannot be found
        """
        if self.args_type == "MODULE_FUNCTION":
            return getattr(self.obj, self.prop)
        elif self.args_type == "FUNCTION":
            return getattr(self.g, self.obj.__name__)
        elif self.args_type == "PURE":
            return getattr(self.pure, "func")
        else:
            ErrorHandler.wrapper_object_not_found_error()