def generate_tar_files(directory_list):
    """Public function that reads a list of local directories and generates tar archives from them"""
    
    tar_file_list = []

    for directory in directory_list:
        if dir_exists(directory):
            _generate_tar(directory)                  # create the tar archive
            tar_file_list.append(directory + '.tar')  # append the tar archive filename to the returned tar_file_list list
        else:
            stderr("The directory '" + directory + "' does not exist and a tar archive could not be created from it.", exit=1)            

    return tar_file_list