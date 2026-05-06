def is_closest_date_parameter(task, param_name):
    """ Return the parameter class of param_name on task. """
    for name, obj in task.get_params():
        if name == param_name:
            return hasattr(obj, 'use_closest_date')
    return False