def map_function(func_str, fw_action_addtion=None,bw_action_addtion=None, alias_func=None):
    ''' Sample usage:
        print map_function('set',alias_func = "ini_items");# -> ini_items
        print map_function('set',fw_action_addtion="action_steps_",bw_action_addtion="_for_upd",alias_func = "ini_items"); # -> action_steps_ini_items_for_upd
        print map_function('set(a=1,b=2,c=Test())',"action_steps_","_for_upd","ini_items");# -> action_steps_ini_items_for_upd(a=1,b=2,c=Test())
        print map_function('set("login",a="good",b=Test())',"action_steps_","_for_upd");# -> action_steps_set_for_upd("login",a="good",b=Test())
    '''
    
    split_action_value = re.compile("^(\w+)(\((.*)\)$)?")
    matched   = split_action_value.match(func_str)    
     
    if matched:
        action = matched.group(1).lower()
        value = matched.group(2)
        #params = matched.group(3)
        
        if alias_func:
            action = alias_func
        if fw_action_addtion:
            action = fw_action_addtion + action        
        if fw_action_addtion:
            action = action + bw_action_addtion
        
        if value:
            return action+value
        else:
            return action