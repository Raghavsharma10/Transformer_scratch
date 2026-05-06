def run(args, features=None):
    """
    Run an ape task.

    Composes task modules out of the selected features and calls the
    task with arguments.

    :param args: list comprised of task name followed by arguments
    :param features: list of features to compose before invoking the task
    """
    features = features or []
    for feature in features:
        tasks_module = get_task_module(feature)
        if tasks_module:
            tasks.superimpose(tasks_module)

    if len(args) < 2 or (len(args) == 2 and args[1] == 'help'):
        tasks.help()
    else:
        taskname = args[1]
        try:
            task = tasks.get_task(taskname, include_helpers=False)
        except TaskNotFound:
            print('Task "%s" not found! Use "ape help" to get usage information.' % taskname)
        else:
            remaining_args = args[2:] if len(args) > 2 else []
            invoke_task(task, remaining_args)