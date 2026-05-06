def uniqueID(size=6, chars=string.ascii_uppercase + string.digits):
    """A quick and dirty way to get a unique string"""
    return ''.join(random.choice(chars) for x in xrange(size))