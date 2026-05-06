def Reference(uri, meaning=None):
    """
    Represents external information, typically original obs data and metadata.

    Args:
        uri(str): Uniform resource identifier for external data, e.g. FITS file.
        meaning(str): The nature of the document referenced, e.g. what
            instrument and filter was used to create the data?
    """
    attrib = {'uri': uri}
    if meaning is not None:
        attrib['meaning'] = meaning
    return objectify.Element('Reference', attrib)