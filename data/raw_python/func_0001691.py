def search_conf_item(start_path, item_type, item_name):
        """ search expected function or variable recursive upward
        @param
            start_path: search start path
            item_type: "function" or "variable"
            item_name: function name or variable name
        e.g.
            search_conf_item('C:/Users/RockFeng/Desktop/s/preference.py','function','test_func')
        """
        dir_path = os.path.dirname(os.path.abspath(start_path))
        target_file = os.path.join(dir_path, "preference.py")
        
        if os.path.isfile(target_file):
            imported_module = ModuleUtils.get_imported_module_from_file(target_file)
            items_dict = ModuleUtils.filter_module(imported_module, item_type)
            if item_name in items_dict:
                return items_dict[item_name]
            else:
                return ModuleUtils.search_conf_item(dir_path, item_type, item_name)
    
        if dir_path == start_path:
            # system root path
            err_msg = "'{}' not found in recursive upward path!".format(item_name)
            if item_type == "function":
                raise p_exception.FunctionNotFound(err_msg)
            else:
                raise p_exception.VariableNotFound(err_msg)
    
        return ModuleUtils.search_conf_item(dir_path, item_type, item_name)