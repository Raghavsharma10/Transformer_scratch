def guess_base_branch():
    # type: (str) -> Optional[str, None]
    """ Try to guess the base branch for the current branch.

    Do not trust this guess. git makes it pretty much impossible to guess
    the base branch reliably so this function implements few heuristics that
    will work on most common use cases but anything a bit crazy will probably
    trip this function.

    Returns:
        Optional[str]: The name of the base branch for the current branch if
        guessable or **None** if can't guess.
    """
    my_branch = current_branch(refresh=True).name

    curr = latest_commit()
    if len(curr.branches) > 1:
        # We're possibly at the beginning of the new branch (currently both
        # on base and new branch).
        other = [x for x in curr.branches if x != my_branch]
        if len(other) == 1:
            return other[0]
        return None
    else:
        # We're on one branch
        parent = curr

        while parent and my_branch in parent.branches:
            curr = parent

            if len(curr.branches) > 1:
                other = [x for x in curr.branches if x != my_branch]
                if len(other) == 1:
                    return other[0]
                return None

            parents = [p for p in curr.parents if my_branch in p.branches]
            num_parents = len(parents)

            if num_parents > 2:
                # More than two parent, give up
                return None
            if num_parents == 2:
                # This is a merge commit.
                for p in parents:
                    if p.branches == [my_branch]:
                        parent = p
                        break
            elif num_parents == 1:
                parent = parents[0]
            elif num_parents == 0:
                parent = None

        return None