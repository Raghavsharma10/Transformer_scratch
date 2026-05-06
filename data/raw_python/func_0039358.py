def randstring(length=1):
    """
    Generate a random string consisting of letters, digits and punctuation

    :type length: integer
    :param length: The length of the generated string.
    """
    charstouse = string.ascii_letters + string.digits + string.punctuation
    newpass = ''
    for _ in range(length):
        newpass += str(charstouse[random.randint(0, len(charstouse) - 1)])
    return newpass