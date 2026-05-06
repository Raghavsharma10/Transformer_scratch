def file_counts(container=None, 
                patterns=None, 
                image_package=None, 
                file_list=None):

    '''file counts will return a list of files that match one or more regular expressions.
    if no patterns is defined, a default of readme is used. All patterns and files are made
    case insensitive.

    Parameters
    ==========
    :param container: if provided, will use container as image. Can also provide
    :param image_package: if provided, can be used instead of container
    :param patterns: one or more patterns (str or list) of files to search for.
    :param diff: the difference between a container and it's parent OS from get_diff
    if not provided, will be generated.

    '''
    if file_list is None:
        file_list = get_container_contents(container, split_delim='\n')['all']

    if patterns == None:
        patterns = 'readme'

    if not isinstance(patterns,list):
        patterns = [patterns]

    count = 0
    for pattern in patterns:
        count += len([x for x in file_list if re.search(pattern.lower(),x.lower())])
    bot.info("Total files matching patterns is %s" %count)
    return count