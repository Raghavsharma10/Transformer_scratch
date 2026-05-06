def min_tasks_per_node(queue_id):
    """
    This function is used when requesting non exclusive use
    as the parallel environment might enforce a minimum number
    of tasks
    """
    parallel_env = queue_id.split(':')[0]
    queue_name = queue_id.split(':')[1]
    tasks = 1
    pe_tasks = tasks
    with os.popen('qconf -sp ' + parallel_env) as f:
        try:
            for line in f:
                if line.split(' ')[0] == 'allocation_rule':
                    # This may throw exception as allocation rule
                    # may not always be an integer
                    pe_tasks = int(re.split('\W+', line)[1])
        except:
            pass

    return max(tasks, pe_tasks)