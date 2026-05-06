def check_for_missed_cleanup():
    """Check for TaskAttempts that were never cleaned up
    """
    if get_setting('PRESERVE_ALL'):
        return
    from api.models.tasks import TaskAttempt
    if get_setting('PRESERVE_ON_FAILURE'):
        for task_attempt in TaskAttempt.objects.filter(
                status_is_running=False).filter(
                    status_is_cleaned_up=False).exclude(
                        status_is_failed=True):
            task_attempt.cleanup()
    else:
        for task_attempt in TaskAttempt.objects.filter(
                status_is_running=False).filter(status_is_cleaned_up=False):
            task_attempt.cleanup()