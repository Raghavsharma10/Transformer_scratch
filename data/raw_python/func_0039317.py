def userinput(prompttext="", times=1):
    """
    Get the input of the user via a universally secure method.

    :type prompttext: string
    :param prompttext: The text to display while receiving the data.

    :type times: integer
    :param times: The amount of times to ask the user. If value is not 1, a list will be returned. Default is 1.

    :return: What the user typed in.
    :rtype: string
    """

    # If times is 1
    if times == 1:
        # Return the result
        return input(str(prompttext))

    # Create new empty list
    inputlist = []

    # For each time in range
    for _ in range(times):
        # Append the result of another input request
        inputlist.append(input(str(prompttext)))

    # Return the final result
    return inputlist