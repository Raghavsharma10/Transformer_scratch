def tarbz(source_directory_path, output_file_full_path, silent=False):
    """ Tars and bzips a directory, preserving as much metadata as possible.
        Adds '.tbz' to the provided output file name. """
    output_directory_path = output_file_full_path.rsplit("/", 1)[0]
    create_folders(output_directory_path)
    # Note: default compression for bzip is supposed to be -9, highest compression.
    full_tar_file_path = output_file_full_path + ".tbz"
    if path.exists(full_tar_file_path):
        raise Exception("%s already exists, aborting." % (full_tar_file_path))
    
    # preserve permissions, create file, use files (not tape devices), preserve
    # access time.  tar is the only program in the universe to use (dstn, src).
    tar_command = ("tar jpcfvC %s %s %s" %
                   (full_tar_file_path, source_directory_path, "./"))
    call(tar_command, silent=silent)
    return full_tar_file_path