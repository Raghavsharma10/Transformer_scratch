def str2hashalgo(description):
    '''Convert the name of a hash algorithm as described in the OATH
       specifications, to a python object handling the digest algorithm
       interface, PEP-xxx.

       :param description
           the name of the hash algorithm, example
       :rtype: a hash algorithm class constructor
    '''
    algo = getattr(hashlib, description.lower(), None)
    if not callable(algo):
        raise ValueError('Unknown hash algorithm %s' % description)
    return algo