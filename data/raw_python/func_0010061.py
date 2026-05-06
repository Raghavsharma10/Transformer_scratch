def generate_random_string(number_of_random_chars=8, character_set=string.ascii_letters):
    """
    Generate a series of random characters.

    Kwargs:
        number_of_random_chars (int) : Number of characters long
        character_set (str): Specify a character set.  Default is ASCII
    """
    return u('').join(random.choice(character_set)
                      for _ in range(number_of_random_chars))