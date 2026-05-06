def get_part_filenames(num_parts=None, start_num=0):
    """Get numbered PART.html filenames."""
    if num_parts is None:
        num_parts = get_num_part_files()
    return ['PART{0}.html'.format(i) for i in range(start_num+1, num_parts+1)]