def wait_for_repo_creation(task_id, retry=30):
    """
    Using polling check if the task finished 
    """
    success_event_types = ("RC_CREATION_SUCCESS", )
    error_event_types = ("RC_REPO_CREATION_ERROR", "RC_REPO_CLONE_ERROR", "RC_CREATION_ERROR")
    while retry > 0:
        bpm_task = get_bpm_task_by_id(task_id)

        if contains_event_type(bpm_task.content.events, success_event_types):
            break

        if contains_event_type(bpm_task.content.events, error_event_types):
            logging.error("Creation of Repository Configuration failed")
            logging.error(bpm_task.content)
            return False

        logging.info("Waiting until Repository Configuration creation task "+str(task_id)+" finishes.")
        time.sleep(10)
        retry -= 1
    return retry > 0