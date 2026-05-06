def compare_filesystems(fs0, fs1, concurrent=False):
    """Compares the two given filesystems.

    fs0 and fs1 are two mounted GuestFS instances
    containing the two disks to be compared.

    If the concurrent flag is True,
    two processes will be used speeding up the comparison on multiple CPUs.

    Returns a dictionary containing files created, removed and modified.

        {'created_files': [<files in fs1 and not in fs0>],
         'deleted_files': [<files in fs0 and not in fs1>],
         'modified_files': [<files in both fs0 and fs1 but different>]}

    """
    if concurrent:
        future0 = concurrent_hash_filesystem(fs0)
        future1 = concurrent_hash_filesystem(fs1)

        files0 = future0.result()
        files1 = future1.result()
    else:
        files0 = hash_filesystem(fs0)
        files1 = hash_filesystem(fs1)

    return file_comparison(files0, files1)