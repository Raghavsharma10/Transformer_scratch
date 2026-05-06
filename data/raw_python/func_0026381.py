def copytree(root_src_dir, root_dst_dir, hardlink=True):
    """Copies a whole directory tree"""

    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            try:
                if os.path.exists(dst_file):
                    if hardlink:
                        hfoslog('Removing frontend link:', dst_file,
                                emitter='BUILDER', lvl=verbose)
                        os.remove(dst_file)
                    else:
                        hfoslog('Overwriting frontend file:', dst_file,
                                emitter='BUILDER', lvl=verbose)

                hfoslog('Hardlinking ', src_file, dst_dir, emitter='BUILDER',
                        lvl=verbose)

                if hardlink:
                    os.link(src_file, dst_file)
                else:
                    copy(src_file, dst_dir)
            except PermissionError as e:
                hfoslog(
                    " No permission to remove/create target %s for "
                    "frontend:" % ('link' if hardlink else 'copy'),
                    dst_dir, e, emitter='BUILDER', lvl=error)
            except Exception as e:
                hfoslog("Error during", 'link' if hardlink else 'copy',
                        "creation:", type(e), e, emitter='BUILDER',
                        lvl=error)

            hfoslog('Done linking', root_dst_dir, emitter='BUILDER',
                    lvl=verbose)