def get_example():
    """Make an example for training and testing.  Outputs a tuple
    (label, features) where label is +1 if capital letters are the majority,
    and -1 otherwise; and features is a list of letters.
    """
    features = random.sample(string.ascii_letters, NUM_SAMPLES)
    num_capitalized = len([ letter for letter in features if letter in string.ascii_uppercase ])
    num_lowercase = len([ letter for letter in features if letter in string.ascii_lowercase ])
    if num_capitalized > num_lowercase:
        label = 1
    else:
        label = -1
    return (label, features)