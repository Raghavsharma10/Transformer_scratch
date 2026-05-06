def convert(data, in_format, out_format, name=None, pretty=False):
    """Converts between two inputted chemical formats.

    Args:
        data: A string representing the chemical file to be converted. If the
            `in_format` is "json", this can also be a Python object
        in_format: The format of the `data` string. Can be "json" or any format
            recognized by Open Babel
        out_format: The format to convert to. Can be "json" or any format
            recognized by Open Babel
        name: (Optional) If `out_format` is "json", will save the specified
            value in a "name" property
        pretty: (Optional) If True and `out_format` is "json", will pretty-
            print the output for human readability
    Returns:
        A string representing the inputted `data` in the specified `out_format`
    """
    # Decide on a json formatter depending on desired prettiness
    dumps = json.dumps if pretty else json.compress

    # Shortcut for avoiding pybel dependency
    if not has_ob and in_format == 'json' and out_format == 'json':
        return dumps(json.loads(data) if is_string(data) else data)
    elif not has_ob:
        raise ImportError("Chemical file format conversion requires pybel.")

    # These use the open babel library to interconvert, with additions for json
    if in_format == 'json':
        mol = json_to_pybel(json.loads(data) if is_string(data) else data)
    elif in_format == 'pybel':
        mol = data
    else:
        mol = pybel.readstring(in_format, data)

    # Infer structure in cases where the input format has no specification
    if not mol.OBMol.HasNonZeroCoords():
        mol.make3D()

    # Make P1 if that's a thing, recalculating bonds in process
    if in_format == 'mmcif' and hasattr(mol, 'unitcell'):
        mol.unitcell.FillUnitCell(mol.OBMol)
        mol.OBMol.ConnectTheDots()
        mol.OBMol.PerceiveBondOrders()

    mol.OBMol.Center()

    if out_format == 'pybel':
        return mol
    elif out_format == 'object':
        return pybel_to_json(mol, name)
    elif out_format == 'json':
        return dumps(pybel_to_json(mol, name))
    else:
        return mol.write(out_format)