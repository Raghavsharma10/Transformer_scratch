def generate_reps(min_reps=3, max_reps=8, total=25, existing=None):
    """Generate 'total' repetitions between 'min_reps' and 'max_reps',
    if existing is given (not None), then some repetitions are already
    drawn.

    Parameters
    ----------
    min_reps
        Lower limit for the repetitions dawing, e.g. 3.
        
    max_reps
        Upper limit for the repetitions dawing, e.g. 8.
        
    total
        The total number of repetitions to return.
        
    existing
        A list of prior reps drawn.

    Returns
    -------
    reps_list
        A list of repetitions between 'min_reps' and 'max_reps', summing to 'total'
        or a number close to it.
    """

    # If no existing repetitions exist, start from empty list
    if existing is None:
        existing = []

    # List of possible rep strings to return
    possible = []

    for _ in range(3):

        # Fill list with randomly drawn reptitions
        created = existing.copy()
        while sum(created) < (total - min_reps):
            generated = random.randint(min_reps, max_reps)
            created.append(generated)

        # Sort and append to the list of lists
        created.sort(reverse=True)
        possible.append(created)

    # Return the list of reps which is closest to the total desired number
    return min(possible, key=lambda l: abs(total - sum(l)))