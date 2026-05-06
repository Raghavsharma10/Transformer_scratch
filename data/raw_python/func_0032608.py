def tar_dir(tarfile, srcdir):
    """ Pack a tar file using all the files in the given srcdir """
    files = os.listdir(srcdir)
    packtar(tarfile, files, srcdir)