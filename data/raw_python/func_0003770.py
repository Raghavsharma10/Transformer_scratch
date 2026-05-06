def dump_cml(f, molecules):
    """Write a list of molecules to a CML file

       Arguments:
        | ``f``  --  a filename of a CML file or a file-like object
        | ``molecules``  --  a list of molecule objects.
    """
    if isinstance(f, str):
        f = open(f, "w")
        close = True
    else:
        close = False
    f.write("<?xml version='1.0'?>\n")
    f.write("<list xmlns='http://www.xml-cml.org/schema'>\n")
    for molecule in molecules:
        _dump_cml_molecule(f, molecule)
    f.write("</list>\n")
    if close:
        f.close()