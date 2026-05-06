def check_presence_of_mandatory_args(args, mandatory_args):
    '''
    Checks whether all mandatory arguments are passed.

    This function aims at methods with many arguments
        which are passed as kwargs so that the order
        in which the are passed does not matter.

    :args: The dictionary passed as args.
    :mandatory_args: A list of keys that have to be
        present in the dictionary.
    :raise: :exc:`~ValueError`
    :returns: True, if all mandatory args are passed. If not,
        an exception is raised.

    '''
    missing_args = []
    for name in mandatory_args:
        if name not in args.keys():
            missing_args.append(name)
    if len(missing_args) > 0:
        raise ValueError('Missing mandatory arguments: '+', '.join(missing_args))
    else:
        return True