def wrap2stub(self, customfunc):
        """
        Wrapping the inspector as a stub based on the type
        Args:
            customfunc: function that replaces the original
        Returns:
            function, the spy wrapper around the customfunc
        """
        if self.args_type == "MODULE_FUNCTION":
            wrapper = Wrapper.wrap_spy(customfunc, self.obj)
            setattr(self.obj, self.prop, wrapper)
        elif self.args_type == "MODULE":
            wrapper = Wrapper.EmptyClass
            setattr(CPSCOPE, self.obj.__name__, wrapper)
        elif self.args_type == "FUNCTION":
            wrapper = Wrapper.wrap_spy(customfunc)
            setattr(CPSCOPE, self.obj.__name__, wrapper)
        elif self.args_type == "PURE":
            wrapper = Wrapper.wrap_spy(customfunc)
            setattr(self.pure, "func", wrapper)
        return wrapper