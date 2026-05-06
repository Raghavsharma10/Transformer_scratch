def extract_classifier_and_extension(pkg_name, filename):
    """
    Returns a PEP425-compliant classifier (or 'py2.py3-none-any' if it cannot be extracted),
    and the file extension
    TODO: return a classifier 3-members namedtuple instead of a single string
    """
    basename, _, extension = filename.rpartition('.')
    if extension == 'gz' and filename.endswith('.tar.gz'):
        extension = 'tar.gz'
        basename = filename[:-7]
    if basename == pkg_name or basename[len(pkg_name)] != '-':
        return 'py2.py3-none-any', extension
    basename = basename[len(pkg_name)+1:]
    classifier_parts = basename.split('-')
    if len(classifier_parts) < 3:
        return 'py2.py3-none-any', extension
    if len(classifier_parts) == 3:
        _, _, classifier_parts[0] = classifier_parts[0].rpartition('.')
    return '-'.join(classifier_parts[-3:]), extension