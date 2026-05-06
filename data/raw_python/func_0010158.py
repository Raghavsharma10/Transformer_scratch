def get_num_part_files():
    """Get the number of PART.html files currently saved to disk."""
    num_parts = 0
    for filename in os.listdir(os.getcwd()):
        if filename.startswith('PART') and filename.endswith('.html'):
            num_parts += 1
    return num_parts