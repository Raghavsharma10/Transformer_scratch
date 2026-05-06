def org_task(task_key, lock_timeout=None):
    """
    Decorator to create an org task.
    :param task_key: the task key used for state storage and locking, e.g. 'do-stuff'
    :param lock_timeout: the lock timeout in seconds
    """

    def _org_task(task_func):
        def _decorator(org_id):
            org = apps.get_model("orgs", "Org").objects.get(pk=org_id)
            maybe_run_for_org(org, task_func, task_key, lock_timeout)

        return shared_task(wraps(task_func)(_decorator))

    return _org_task