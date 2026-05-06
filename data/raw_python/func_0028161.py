def variable_status(code: str,
                    exclude_variable: Union[set, None] = None,
                    jsonable_parameter: bool = True) -> tuple:
    """
    Find the possible parameters and "global" variables from a python code.

    This is achieved by parsing the abstract syntax tree.

    Parameters
    ----------
    code : str
        Input code as string.
    exclude_variable : set, None, optional
        Variable to exclude.
    jsonable_parameter: bool, True, optional
        Consider only jsonable parameter

    Returns
    -------
    tuple
        (a set of possible parameter, a set of parameter to exclude, a dictionary of possible parameter )

        A variable is a possible parameter if 1) it is not in the input exclude_variable,
        2) the code contains only assignments, and 3) it is used only to bound objects.

        The set of parameter to exclude is the union of the input exclude_variable and all names that looks like a global variable.
        The dictionary of possible parameter {parameter name, parameter value} is available only if jsonable_parameter is True. 




    >>> variable_status("a=3")
    ({'a'}, {'a'}, {'a': 3})
    >>> variable_status("a=3",jsonable_parameter=False)
    ({'a'}, {'a'}, {})
    >>> variable_status("a += 1")
    (set(), {'a'}, {})
    >>> variable_status("def f(x,y=3):\\n\\t pass")
    (set(), {'f'}, {})
    >>> variable_status("class C(A):\\n\\t pass")
    (set(), {'C'}, {})
    >>> variable_status("import f")
    (set(), {'f'}, {})
    >>> variable_status("import f as g")
    (set(), {'g'}, {})
    >>> variable_status("from X import f")
    (set(), {'f'}, {})
    >>> variable_status("from X import f as g")
    (set(), {'g'}, {})
    """
    if exclude_variable is None:
        exclude_variable = set()
    else:
        exclude_variable = copy.deepcopy(exclude_variable)
    root = ast.parse(code)
    store_variable_name = set()
    assign_only = True
    dict_parameter={}

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Assign):
            for assign_node in ast.walk(node):
                if isinstance(assign_node, ast.Name):

                    if isinstance(assign_node.ctx, ast.Store):
                        if jsonable_parameter is False:
                            store_variable_name |= {assign_node.id}
                    else:
                        exclude_variable |= {assign_node.id}

            _is_literal_eval,_value=is_literal_eval(node.value)

            if jsonable_parameter is True:
                for assign_node in ast.iter_child_nodes(node):
                    if isinstance(assign_node, ast.Tuple):
                        i=0
                        for assign_tuple_node in ast.iter_child_nodes(assign_node):
                            if isinstance(assign_tuple_node, ast.Name):

                                if isinstance(_value,(collections.Iterable)) and is_jsonable(_value[i]) and _is_literal_eval:
                                    dict_parameter[assign_tuple_node.id]=_value[i]                                
                                    store_variable_name |= {assign_tuple_node.id} 
                                else:
                                    exclude_variable |= {assign_tuple_node.id}
                                i += 1
                    else:
                        if isinstance(assign_node, ast.Name):
                            if is_jsonable(_value) and _is_literal_eval:
                                dict_parameter[assign_node.id]=_value
                                store_variable_name |= {assign_node.id} 
                            else:
                                exclude_variable |= {assign_node.id}

        elif isinstance(node, ast.AugAssign):
            for assign_node in ast.walk(node):
                if isinstance(assign_node, ast.Name):
                    exclude_variable |= {assign_node.id}
        # class and function
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assign_only = False
            exclude_variable |= {node.name}
        # import
        elif isinstance(node, ast.Import):
            assign_only = False
            for node1 in ast.iter_child_nodes(node):
                if node1.asname is not None:
                    exclude_variable |= {node1.asname}
                else:
                    exclude_variable |= {node1.name}
        # import from
        elif isinstance(node, ast.ImportFrom):
            assign_only = False
            for node1 in ast.iter_child_nodes(node):
                if node1.asname is not None:
                    exclude_variable |= {node1.asname}
                else:
                    exclude_variable |= {node1.name}
        else:
            assign_only = False
    if assign_only is True:
        possible_parameter = store_variable_name-exclude_variable
        if jsonable_parameter is True:
            dict_parameter = {k:dict_parameter[k] for k in possible_parameter}
        return (possible_parameter, store_variable_name | exclude_variable, dict_parameter)
    return set(), store_variable_name | exclude_variable, {}