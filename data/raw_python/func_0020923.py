def friends_description(baseURLs):
    """Generate the friends element for the identify response.

    The friends container is recommended for use by repositories
    to list confederate repositories.
    For the schema definition visit:
    http://www.openarchives.org/OAI/2.0/guidelines-friends.htm
    """
    friends = Element(etree.QName(NS_FRIENDS[None], 'friends'),
                      nsmap=NS_FRIENDS)
    friends.set(etree.QName(ns['xsi'], 'schemaLocation'),
                '{0} {1}'.format(FRIENDS_SCHEMA_LOCATION,
                                 FRIENDS_SCHEMA_LOCATION_XSD))
    for baseURL in baseURLs:
        friends.append(E('baseURL', baseURL))
    return etree.tostring(friends, pretty_print=True)