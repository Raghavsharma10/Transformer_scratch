def wrap2spy(self):
        """
        Wrapping the inspector as a spy based on the type
        """
        if self.args_type == "MODULE_FUNCTION":
            self.orig_func = deepcopy(getattr(self.obj, self.prop))
            setattr(self.obj, self.prop, Wrapper.wrap_spy(getattr(self.obj, self.prop)))
        elif self.args_type == "MODULE":
            setattr(self.obj, "__SINONLOCK__", True)
        elif self.args_type == "FUNCTION":
            self.orig_func = deepcopy(getattr(CPSCOPE, self.obj.__name__))
            setattr(CPSCOPE, self.obj.__name__,
                    Wrapper.wrap_spy(getattr(CPSCOPE, self.obj.__name__)))
        elif self.args_type == "PURE":
            self.orig_func = deepcopy(getattr(self.pure, "func"))
            setattr(self.pure, "func", Wrapper.wrap_spy(getattr(self.pure, "func")))