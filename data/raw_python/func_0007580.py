def guess_tags(filename):
    """
    Function to get potential tags for files using the file names.

    :param filename: This field is the name of file.
    """
    tags = []
    stripped_filename = strip_zip_suffix(filename)
    if stripped_filename.endswith('.vcf'):
        tags.append('vcf')
    if stripped_filename.endswith('.json'):
        tags.append('json')
    if stripped_filename.endswith('.csv'):
        tags.append('csv')
    return tags