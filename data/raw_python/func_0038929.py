def maybe_run_for_org(org, task_func, task_key, lock_timeout):
    """
    Runs the given task function for the specified org provided it's not already running
    :param org: the org
    :param task_func: the task function
    :param task_key: the task key
    :param lock_timeout: the lock timeout in seconds
    """
    r = get_redis_connection()

    key = TaskState.get_lock_key(org, task_key)

    if r.get(key):
        logger.warning("Skipping task %s for org #%d as it is still running" % (task_key, org.id))
    else:
        with r.lock(key, timeout=lock_timeout):
            state = org.get_task_state(task_key)
            if state.is_disabled:
                logger.info("Skipping task %s for org #%d as is marked disabled" % (task_key, org.id))
                return

            logger.info("Started task %s for org #%d..." % (task_key, org.id))

            prev_started_on = state.last_successfully_started_on
            this_started_on = timezone.now()

            state.started_on = this_started_on
            state.ended_on = None
            state.save(update_fields=("started_on", "ended_on"))

            num_task_args = len(inspect.getargspec(task_func).args)

            try:
                if num_task_args == 3:
                    results = task_func(org, prev_started_on, this_started_on)
                elif num_task_args == 1:
                    results = task_func(org)
                else:
                    raise ValueError("Task signature must be foo(org) or foo(org, since, until)")  # pragma: no cover

                state.ended_on = timezone.now()
                state.last_successfully_started_on = this_started_on
                state.last_results = json.dumps(results)
                state.is_failing = False
                state.save(update_fields=("ended_on", "last_successfully_started_on", "last_results", "is_failing"))

                logger.info("Finished task %s for org #%d with result: %s" % (task_key, org.id, json.dumps(results)))

            except Exception as e:
                state.ended_on = timezone.now()
                state.last_results = None
                state.is_failing = True
                state.save(update_fields=("ended_on", "last_results", "is_failing"))

                logger.exception("Task %s for org #%d failed" % (task_key, org.id))
                raise e