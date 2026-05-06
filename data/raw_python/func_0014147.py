def convert(json_input, build_direction="LEFT_TO_RIGHT", table_attributes=None):
    """
    Converts JSON to HTML Table format.

    Parameters
    ----------
    json_input : dict
        JSON object to convert into HTML.
    build_direction : {"TOP_TO_BOTTOM", "LEFT_TO_RIGHT"}
        String denoting the build direction of the table. If ``"TOP_TO_BOTTOM"`` child
        objects will be appended below parents, i.e. in the subsequent row. If ``"LEFT_TO_RIGHT"``
        child objects will be appended to the right of parents, i.e. in the subsequent column.
        Default is ``"LEFT_TO_RIGHT"``.
    table_attributes : dict, optional
        Dictionary of ``(key, value)`` pairs describing attributes to add to the table. 
        Each attribute is added according to the template ``key="value". For example, 
        the table ``{ "border" : 1 }`` modifies the generated table tags to include 
        ``border="1"`` as an attribute. The generated opening tag would look like 
        ``<table border="1">``. Default is ``None``.

    Returns
    -------
    str
        String of converted HTML.

    An example usage is shown below:

    >>> json_object = {"key" : "value"}
    >>> build_direction = "TOP_TO_BOTTOM"
    >>> table_attributes = {"border" : 1}
    >>> html = convert(json_object, build_direction=build_direction, table_attributes=table_attributes)
    >>> print(html)
    "<table border="1"><tr><th>key</th><td>value</td></tr></table>"

    """
    json_converter = JsonConverter(build_direction=build_direction, table_attributes=table_attributes)
    return json_converter.convert(json_input)