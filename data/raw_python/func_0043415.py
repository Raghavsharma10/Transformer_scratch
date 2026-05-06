def collect_static_files(src_map, dst):
    """
    Collect all static files and move them into a temporary location.

    This is very similar to the ``collectstatic`` command.
    """
    for rel_src, abs_src in src_map.iteritems():
        abs_dst = os.path.join(dst, rel_src)
        copy_file(abs_src, abs_dst)