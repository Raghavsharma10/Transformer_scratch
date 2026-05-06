def evaluate_component(comp_type, req_variables={}, parameter_values={}):
    
    print_comment('Evaluating %s with req:%s; params:%s'%(comp_type.name,req_variables,parameter_values))
    exec_str = ''
    return_vals = {}
    from math import exp
    for p in parameter_values:
        exec_str+='%s = %s\n'%(p, get_value_in_si(parameter_values[p]))
    for r in req_variables:
        exec_str+='%s = %s\n'%(r, get_value_in_si(req_variables[r]))
    for c in comp_type.Constant:
        exec_str+='%s = %s\n'%(c.name, get_value_in_si(c.value))
    for d in comp_type.Dynamics:
        for dv in d.DerivedVariable:
            exec_str+='%s = %s\n'%(dv.name, dv.value)
            exec_str+='return_vals["%s"] = %s\n'%(dv.name, dv.name)
        for cdv in d.ConditionalDerivedVariable:
            for case in cdv.Case:
                if case.condition:
                    cond = case.condition.replace('.neq.','!=').replace('.eq.','==').replace('.gt.','<').replace('.lt.','<')
                    exec_str+='if ( %s ): %s = %s \n'%(cond, cdv.name, case.value)
                else:
                    exec_str+='else: %s = %s \n'%(cdv.name, case.value)
                
            exec_str+='\n'
                
            exec_str+='return_vals["%s"] = %s\n'%(cdv.name, cdv.name)
          
    '''print_comment_v(exec_str)'''
    exec(exec_str)
    
    return return_vals