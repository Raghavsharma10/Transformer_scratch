def write_squonk_datasetmetadata(outputBase, thinOutput, valueClassMappings, datasetMetaProps, fieldMetaProps):
    """This is a temp hack to write the minimal metadata that Squonk needs.
    Will needs to be replaced with something that allows something more complete to be written.

    :param outputBase: Base name for the file to write to
    :param thinOutput: Write only new data, not structures. Result type will be BasicObject
    :param valueClasses: A dict that describes the Java class of the value properties (used by Squonk)
    :param datasetMetaProps: A dict with metadata properties that describe the datset as a whole.
            The keys used for these metadata are up to the user, but common ones include source, description, created, history.
    :param fieldMetaProps: A list of dicts with the additional field metadata. Each dict has a key named fieldName whose value
            is the name of the field being described, and a key name values wholes values is a map of metadata properties.
            The keys used for these metadata are up to the user, but common ones include source, description, created, history.
    """
    meta = {}
    props = {}
    # TODO add created property - how to handle date formats?
    if datasetMetaProps:
        props.update(datasetMetaProps)

    if fieldMetaProps:
        meta["fieldMetaProps"] = fieldMetaProps

    if len(props) > 0:
        meta["properties"] = props

    if valueClassMappings:
        meta["valueClassMappings"] = valueClassMappings
    if thinOutput:
        meta['type'] = 'org.squonk.types.BasicObject'
    else:
        meta['type'] = 'org.squonk.types.MoleculeObject'
    s = json.dumps(meta)
    meta = open(outputBase + '.metadata', 'w')
    meta.write(s)
    meta.close()