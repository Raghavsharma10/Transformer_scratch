def get_tags(container=None,
             search_folders=None,
             file_list=None,
             return_unique=True):

    '''get tags will return a list of tags that describe the software in an image,
    meaning inside of a paricular folder. If search_folder is not defined, uses lib
    :param container: if provided, will use container as image. Can also provide
    :param image_package: if provided, can be used instead of container
    :param search_folders: specify one or more folders to look for tags 
    :param file_list: the list of files
    :param return_unique: return unique files in folders. Default True.
    Default is 'bin'

    ::notes
  
    The algorithm works as follows:
      1) first compare package to set of base OS (provided with shub)
      2) subtract the most similar os from image, leaving "custom" files
      3) organize custom files into dict based on folder name
      4) return search_folders as tags

    '''
    if file_list is None:
        file_list = get_container_contents(container, split_delim='\n')['all']

    if search_folders == None:
        search_folders = 'bin'

    if not isinstance(search_folders,list):
        search_folders = [search_folders]

    tags = []
    for search_folder in search_folders:
        for file_name in file_list:
            if search_folder in file_name:
                tags.append(file_name)

    if return_unique == True:
        tags = list(set(tags))
    return tags