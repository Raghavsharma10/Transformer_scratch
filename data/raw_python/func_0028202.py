def possible_parameter(nb, jsonable_parameter=True, end_cell_index=None):
    """
    Find the possible parameters from a jupyter notebook (python3 only).

    The possible parameters are obtained by parsing the abstract syntax tree of
    the python code generated from the jupyter notebook.

    For a jupuyter notebook, a variable can be a possible parameter if:
        - it is defined in a cell that contains only comments or assignments,
        - its name is not used in the current cell beside the assignment nor previously.


    Parameters
    ----------
    nb : str, nbformat.notebooknode.NotebookNode
        Jupyter notebook path or its content as a NotebookNode object.
    jsonable_parameter: bool, optional
        Consider only jsonable parameters.
    end_cell_index : int, optional
        End cell index used to slice the notebook in finding the possible parameters.

    Returns
    -------
    list[collections.namedtuple]
        If jsonable_parameter is true the fields are ('name','value','cell_index'), otherwise ('name', 'cell_index').
        The list is ordered by the name of the parameters.
    """
    jh = _JupyterNotebookHelper(nb, jsonable_parameter, end_cell_index)

    if jsonable_parameter is True:
        PossibleParameter=collections.namedtuple('PossibleParameter',['name','value','cell_index'])
    else:
        PossibleParameter=collections.namedtuple('PossibleParameter',['name', 'cell_index'])

    res=[]
    for name, cell_index in jh.param_cell_index.items():
        if jsonable_parameter is True:
            res.append(PossibleParameter(name=name,value=jh.param_value[name],cell_index=cell_index))
        else:
            res.append(PossibleParameter(name=name,cell_index=cell_index))

    return sorted(res, key = lambda x: (x.name))