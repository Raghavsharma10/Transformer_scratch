def messages(count, size):
    '''Generator for count messages of the provided size'''
    import string
    # Make sure we have at least 'size' letters
    letters = islice(cycle(chain(string.lowercase, string.uppercase)), size)
    return islice(cycle(''.join(l) for l in permutations(letters, size)), count)