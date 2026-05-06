def get_file(db_folder, file_name):
    """Glob for the poor."""
    if not os.path.isdir(db_folder):
        return
    file_name = file_name.lower().strip()
    for cand_name in os.listdir(db_folder):
        if cand_name.lower().strip() == file_name:
            return os.path.join(db_folder, cand_name)