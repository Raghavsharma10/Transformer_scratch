def filter_module(module, filter_type):
        """ filter functions or variables from import module
        @params
            module: imported module
            filter_type: "function" or "variable"
        """
        filter_type = ModuleUtils.is_function if filter_type == "function" else ModuleUtils.is_variable
        module_functions_dict = dict(filter(filter_type, vars(module).items()))
        return module_functions_dict