def generate_molecule_object_dict(source, format, values):
    """Generate a dictionary that represents a Squonk MoleculeObject when
    written as JSON

    :param source: Molecules in molfile or smiles format
    :param format: The format of the molecule. Either 'mol' or 'smiles'
    :param values: Optional dict of values (properties) for the MoleculeObject
    """
    m = {"uuid": str(uuid.uuid4()), "source": source, "format": format}
    if values:
        m["values"] = values
    return m