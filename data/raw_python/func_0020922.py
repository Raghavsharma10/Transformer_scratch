def oai_identifier_description(scheme, repositoryIdentifier,
                               delimiter, sampleIdentifier):
    """Generate the oai-identifier element for the identify response.

    The OAI identifier format is intended to provide persistent resource
    identifiers for items in repositories that implement OAI-PMH.
    For the full specification and schema definition visit:
    http://www.openarchives.org/OAI/2.0/guidelines-oai-identifier.htm
    """
    oai_identifier = Element(etree.QName(NS_OAI_IDENTIFIER[None],
                             'oai_identifier'),
                             nsmap=NS_OAI_IDENTIFIER)
    oai_identifier.set(etree.QName(ns['xsi'], 'schemaLocation'),
                       '{0} {1}'.format(OAI_IDENTIFIER_SCHEMA_LOCATION,
                                        OAI_IDENTIFIER_SCHEMA_LOCATION_XSD))
    oai_identifier.append(E('scheme', scheme))
    oai_identifier.append(E('repositoryIdentifier', repositoryIdentifier))
    oai_identifier.append(E('delimiter', delimiter))
    oai_identifier.append(E('sampleIdentifier', sampleIdentifier))
    return etree.tostring(oai_identifier, pretty_print=True)