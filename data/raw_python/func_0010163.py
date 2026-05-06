def remove_part_images(filename):
    """Remove PART(#)_files directory containing images from disk."""
    dirname = '{0}_files'.format(os.path.splitext(filename)[0])
    if os.path.exists(dirname):
        shutil.rmtree(dirname)