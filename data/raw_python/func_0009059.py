def get_container_contents(container, split_delim=None):
    '''get_container_contents will return a list of folders and or files
    for a container. The environmental variable SINGULARITY_HUB being set
    means that container objects are referenced instead of packages
    :param container: the container to get content for
    :param gets: a list of file names to return, without parent folders
    :param split_delim: if defined, will split text by split delimiter
    '''

    # We will look for everything in guts, then return it
    guts = dict()

    SINGULARITY_HUB = os.environ.get('SINGULARITY_HUB',"False")

    # Visualization deployed local or elsewhere
    if SINGULARITY_HUB == "False":
        file_obj,tar = get_image_tar(container)     
        guts = extract_guts(image_path=container, tar=tar)
        delete_image_tar(file_obj, tar)

    # Visualization deployed by singularity hub
    else:   
        
        # user has provided a package, but not a container
        if container == None:
            guts = load_package(image_package,get=gets)

        # user has provided a container, but not a package
        else:
            for sfile in container.files:
                for gut_key in gets:        
                    if os.path.basename(sfile['name']) == gut_key:
                        if split_delim == None:
                            guts[gut_key] = requests.get(sfile['mediaLink']).text
                        else:
                            guts[gut_key] = requests.get(sfile['mediaLink']).text.split(split_delim)

    return guts