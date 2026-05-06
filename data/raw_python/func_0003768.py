def load_cml(cml_filename):
    """Load the molecules from a CML file

       Argument:
        | ``cml_filename``  --  The filename of a CML file.

       Returns a list of molecule objects with optional molecular graph
       attribute and extra attributes.
    """
    parser = make_parser()
    parser.setFeature(feature_namespaces, 0)
    dh = CMLMoleculeLoader()
    parser.setContentHandler(dh)
    parser.parse(cml_filename)
    return dh.molecules