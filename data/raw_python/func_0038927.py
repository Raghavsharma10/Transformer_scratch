def trigger_org_task(task_name, queue="celery"):
    """
    Triggers the given org task to be run for all active orgs
    :param task_name: the full task name, e.g. 'myproj.myapp.tasks.do_stuff'
    :param queue: the name of the queue to send org sub-tasks to
    """
    active_orgs = apps.get_model("orgs", "Org").objects.filter(is_active=True)
    for org in active_orgs:
        sig = signature(task_name, args=[org.pk])
        sig.apply_async(queue=queue)

    logger.info("Requested task '%s' for %d active orgs" % (task_name, len(active_orgs)))