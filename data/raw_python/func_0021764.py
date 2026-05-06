def untarbz(source_file_path, output_directory_path, silent=False):
    """ Restores your mongo database backup from a .tbz created using this library.
    This function will ensure that a directory is created at the file path
    if one does not exist already.
    
    If used in conjunction with this library's mongodump operation, the backup
    data will be extracted directly into the provided directory path.
    
    This command will fail if the output directory is not empty as existing files
    with identical names are not overwritten by tar. """
    
    if not path.exists(source_file_path):
        raise Exception("the provided tar file %s does not exist." % (source_file_path))
    
    if output_directory_path[0:1] == "./":
        output_directory_path = path.abspath(output_directory_path)
    if output_directory_path[0] != "/":
        raise Exception("your output directory path must start with '/' or './'; you used: %s"
                        % (output_directory_path))
    create_folders(output_directory_path)
    if listdir(output_directory_path):
        raise Exception("Your output directory isn't empty.  Aborting as "
                        + "exiting files are not overwritten by tar.")
    
    untar_command = ("tar jxfvkCp %s %s --atime-preserve " %
                     (source_file_path, output_directory_path))
    call(untar_command, silent=silent)