def remove_part_files(num_parts=None):
    """Remove PART(#).html files and image directories from disk."""
    filenames = get_part_filenames(num_parts)
    for filename in filenames:
        remove_part_images(filename)
        remove_file(filename)