def blksize(path):
    """
    Get optimal file system buffer size (in bytes) for I/O calls.
    """
    diskfreespace = win32file.GetDiskFreeSpace
    dirname = os.path.dirname(fullpath(path))
    try:
        cluster_sectors, sector_size = diskfreespace(dirname)[:2]
        size = cluster_sectors * sector_size

    except win32file.error as e:
        if e.winerror != winerror.ERROR_NOT_READY:
            raise
        sleep(3)
        size = blksize(dirname)

    return size