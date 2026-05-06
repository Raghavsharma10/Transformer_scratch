def eprints_description(metadataPolicy, dataPolicy,
                        submissionPolicy=None, content=None):
    """Generate the eprints element for the identify response.

    The eprints container is used by the e-print community to describe
    the content and policies of repositories.
    For the full specification and schema definition visit:
    http://www.openarchives.org/OAI/2.0/guidelines-eprints.htm
    """
    eprints = Element(etree.QName(NS_EPRINTS[None], 'eprints'),
                      nsmap=NS_EPRINTS)
    eprints.set(etree.QName(ns['xsi'], 'schemaLocation'),
                '{0} {1}'.format(EPRINTS_SCHEMA_LOCATION,
                                 EPRINTS_SCHEMA_LOCATION_XSD))
    if content:
        contentElement = etree.Element('content')
        for key, value in content.items():
            contentElement.append(E(key, value))
        eprints.append(contentElement)

    metadataPolicyElement = etree.Element('metadataPolicy')
    for key, value in metadataPolicy.items():
        metadataPolicyElement.append(E(key, value))
        eprints.append(metadataPolicyElement)

    dataPolicyElement = etree.Element('dataPolicy')
    for key, value in dataPolicy.items():
        dataPolicyElement.append(E(key, value))
        eprints.append(dataPolicyElement)

    if submissionPolicy:
        submissionPolicyElement = etree.Element('submissionPolicy')
        for key, value in submissionPolicy.items():
            submissionPolicyElement.append(E(key, value))
        eprints.append(submissionPolicyElement)
    return etree.tostring(eprints, pretty_print=True)