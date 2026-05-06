def copy_wildcard(src_folder, dst_folder, glob):
    """copy
    """
    create_dir(dst_folder)
    for sname in iglob(os.path.join(src_folder, glob)):
        rname = os.path.relpath(sname, src_folder)
        dname = os.path.join(dst_folder, rname)
        create_dir(dname)
        shutil.copy(sname, dname)