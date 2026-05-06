def sum_revision_numbers(arguments):
    """
    Sum revision numbers of multiple repository/revision pairs.

    :param arguments: A list of strings with repository names and revision
                      strings.
    :returns: A single integer containing the summed revision numbers.

    This is useful when you're building a package based on revisions from
    multiple VCS repositories. By taking changes in all repositories into
    account when generating version numbers you can make sure that your version
    number is bumped with every single change.
    """
    arguments = list(arguments)
    if len(arguments) % 2 != 0:
        raise ValueError("Please provide an even number of arguments! (one or more repository/revision pairs)")
    summed_revision_number = 0
    while arguments:
        repository = coerce_repository(arguments.pop(0))
        summed_revision_number += repository.find_revision_number(arguments.pop(0))
    return summed_revision_number