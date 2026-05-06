def assess_content(member,file_filter):
    '''Determine if the filter wants the file to be read for content.
    In the case of yes, we would then want to add the content to the
    hash and not the file object.
    '''
    member_path = member.name.replace('.','',1)

    if len(member_path) == 0:
        return False

    # Does the filter skip it explicitly?
    if "skip_files" in file_filter:
        if member_path in file_filter['skip_files']:
            return False

    if "assess_content" in file_filter:
        if member_path in file_filter['assess_content']:
            return True
    return False