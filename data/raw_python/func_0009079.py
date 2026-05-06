def include_file(member,file_filter):
    '''include_file will look at a path and determine
    if it matches a regular expression from a level
    '''
    member_path = member.name.replace('.','',1)

    if len(member_path) == 0:
        return False

    # Does the filter skip it explicitly?
    if "skip_files" in file_filter:
        if member_path in file_filter['skip_files']:
            return False

    # Include explicitly?
    if "include_files" in file_filter:
        if member_path in file_filter['include_files']:
            return True

    # Regular expression?
    if "regexp" in file_filter:
        if re.search(file_filter["regexp"],member_path):
            return True
    return False