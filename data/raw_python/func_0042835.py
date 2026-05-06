def __set_type(self, obj, prop):
        """
        Triage type based on arguments
        Here are four types of base: PURE, MODULE, MODULE_FUNCTION, FUNCTION
        Args:
            obj: None, FunctionType, ModuleType, Class, Instance
            prop: None, string
        """
        if TypeHandler.is_pure(obj, prop):
            self.args_type = "PURE"
            self.pure = SinonBase.Pure()
            setattr(self.pure, "func", Wrapper.empty_function)
            self.orig_func = None
        elif TypeHandler.is_module_function(obj, prop):
            self.args_type = "MODULE_FUNCTION"
            self.orig_func = None
        elif TypeHandler.is_function(obj):
            self.args_type = "FUNCTION"
            self.orig_func = None
        elif TypeHandler.is_module(obj):
            self.args_type = "MODULE"
        elif TypeHandler.is_instance(obj):
            obj = obj.__class__
            self.args_type = "MODULE"