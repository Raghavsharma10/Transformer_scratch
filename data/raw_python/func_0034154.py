def find_target_migration_file(database=DEFAULT_DB_ALIAS, changelog_file=None):
    """Finds best matching target migration file"""

    if not database:
        database = DEFAULT_DB_ALIAS

    if not changelog_file:
        changelog_file = get_changelog_file_for_database(database)

    try:
        doc = minidom.parse(changelog_file)
    except ExpatError as ex:
        raise InvalidChangelogFile(
                'Could not parse XML file %s: %s' % (changelog_file, ex))

    try:
        dbchglog = doc.getElementsByTagName('databaseChangeLog')[0]
    except IndexError:
        raise InvalidChangelogFile(
            'Missing <databaseChangeLog> node in file %s' % (
                                                    changelog_file))
    else:
        nodes = list(filter(lambda x: x.nodeType is x.ELEMENT_NODE,
                            dbchglog.childNodes))
        if not nodes:
            return changelog_file

        last_node = nodes[-1]

        if last_node.tagName == 'include':
            last_file = last_node.attributes.get('file').firstChild.data
            return find_target_migration_file(
                    database=database, changelog_file=last_file)
        else:
            return changelog_file