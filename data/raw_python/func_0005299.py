def create_simple_writer(outputDef, defaultOutput, outputFormat, fieldNames,
                         compress=True, valueClassMappings=None,
                         datasetMetaProps=None, fieldMetaProps=None):
    """Create a simple writer suitable for writing flat data
    e.g. as BasicObject or TSV."""

    if not outputDef:
        outputBase = defaultOutput
    else:
        outputBase = outputDef

    if outputFormat == 'json':
        write_squonk_datasetmetadata(outputBase, True, valueClassMappings,
                                     datasetMetaProps, fieldMetaProps)
        return BasicObjectWriter(open_output(outputDef, 'data', compress)), outputBase

    elif outputFormat == 'tsv':
        return TsvWriter(open_output(outputDef, 'tsv', compress), fieldNames), outputBase

    else:
        raise ValueError("Unsupported format: " + outputFormat)