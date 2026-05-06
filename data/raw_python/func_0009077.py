def extension_counts(container=None, file_list=None, return_counts=True):
    '''extension counts will return a dictionary with counts of file extensions for
    an image.
    :param container: if provided, will use container as image. Can also provide
    :param image_package: if provided, can be used instead of container
    :param file_list: the complete list of files
    :param return_counts: return counts over dict with files. Default True
    '''
    if file_list is None:
        file_list = get_container_contents(container, split_delim='\n')['all']

    extensions = dict()
    for item in file_list:
        filename,ext = os.path.splitext(item)
        if ext == '':
            if return_counts == False:
                extensions = update_dict(extensions,'no-extension',item)
            else:
                extensions = update_dict_sum(extensions,'no-extension')
        else:
            if return_counts == False:
                extensions = update_dict(extensions,ext,item)
            else:
                extensions = update_dict_sum(extensions,ext)

    return extensions