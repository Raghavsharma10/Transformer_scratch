def sample_less_than_condition(choices_in, condition):
    """Creates a random sample from choices without replacement, subject to the
    condition that each element of the output is greater than the corresponding
    element of the condition array.

    condition should be in ascending order.
    """
    output = np.zeros(min(condition.shape[0], choices_in.shape[0]))
    choices = copy.deepcopy(choices_in)
    for i, _ in enumerate(output):
        # randomly select one of the choices which meets condition
        avail_inds = np.where(choices < condition[i])[0]
        selected_ind = np.random.choice(avail_inds)
        output[i] = choices[selected_ind]
        # remove the chosen value
        choices = np.delete(choices, selected_ind)
    return output