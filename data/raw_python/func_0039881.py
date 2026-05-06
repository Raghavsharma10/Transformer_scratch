def invoke_task(task, args):
    """
    Parse args and invoke task function.

    :param task: task function to invoke
    :param args: arguments to the task (list of str)
    :return: result of task function
    :rtype: object
    """
    parser, proxy_args = get_task_parser(task)
    if proxy_args:
        return task(*args)
    else:
        pargs = parser.parse_args(args)
        return task(**vars(pargs))