def unwrap(self):
        """
        Unwrapping the inspector based on the type
        """
        if self.args_type == "MODULE_FUNCTION":
            setattr(self.obj, self.prop, self.orig_func)
        elif self.args_type == "MODULE":
            delattr(self.obj, "__SINONLOCK__")
        elif self.args_type == "FUNCTION":
            setattr(CPSCOPE, self.obj.__name__, self.orig_func)
        elif self.args_type == "PURE":
            setattr(self.pure, "func", self.orig_func)