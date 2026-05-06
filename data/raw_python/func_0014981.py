def get_default_if():
    """ Returns the default interface """
    f = open ('/proc/net/route', 'r')
    for line in f:
        words = line.split()
        dest = words[1]
        try:
            if (int (dest) == 0):
                interf = words[0]
                break
        except ValueError:
            pass
    return interf