def generate_sources_zip(milestone_id=None, output=None):
    """
    Generate a sources archive for given milestone id. 
    """
    if not is_input_valid(milestone_id, output):
        logging.error("invalid input")
        return 1
    create_work_dir(output)
    download_sources_artifacts(milestone_id, output)
    create_zip(output)