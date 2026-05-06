def get_parameters(self):
        """stub"""
        if not self.is_parameterized():
            raise IllegalState()
        if not self.get_text('python_script'):
            return dict()
        import imp
        from types import ModuleType, FunctionType
        script_module = imp.new_module('script_module')
        exec(self.get_text('python_script').text, script_module.__dict__)
        params = dict()
        for attr in dir(script_module):
            if (not isinstance(getattr(script_module, attr), ModuleType) and
                    not isinstance(getattr(script_module, attr), FunctionType) and
                    not attr.startswith('__')):
                params[attr] = getattr(script_module, attr)
        return params