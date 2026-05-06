def wordvalue(word):
    """
    Get the value of each letter of a string's position in the alphabet added up

    :type word: string
    :param word: The word to find the value of
    """

    # Set total to 0
    total = 0

    # For each character of word
    for i in enumerate(word):
        # Add it's letter value to total
        total += letternum(word[i[0]])

    # Return the final value
    return total