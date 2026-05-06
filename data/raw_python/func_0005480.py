def strkey(val, chaffify=1, keyspace=string.ascii_letters + string.digits):
    """ Converts integers to a sequence of strings, and reverse.
        This is not intended to obfuscate numbers in any kind of
        cryptographically secure way, in fact it's the opposite. It's
        for predictable, reversable, obfuscation. It can also be used to
        transform a random bit integer to a string of the same bit
        length.

        @val: #int or #str
        @chaffify: #int multiple to avoid 0=a, 1=b, 2=c, ... obfuscates the
            ordering
        @keyspace: #str allowed output chars

        -> #str if @val is #int, #int if @val is #str

        ..
            from vital.security import strkey

            strkey(0, chaffify=1)
            # -> b
            strkey(0, chaffify=4)
            # -> e
            strkey(90000000000050500502200302035023)
            # -> 'f3yMpJQUazIZHp1UO7k'
            strkey('f3yMpJQUazIZHp1UO7k')
            # -> 90000000000050500502200302035023
            strkey(2000000, chaffify=200000000000)
            # -> 'DIaqtyo2sC'
        ..
    """
    chaffify = chaffify or 1
    keylen = len(keyspace)
    try:
        # INT TO STRING
        if val < 0:
            raise ValueError("Input value must be greater than -1.")

        # chaffify the value
        val = val * chaffify

        if val == 0:
            return keyspace[0]

        # output the new string value
        out = []
        out_add = out.append
        
        while val > 0:
            val, digit = divmod(val, keylen)
            out_add(keyspace[digit])

        return "".join(out)[::-1]
    except TypeError:
        # STRING TO INT
        out = 0
        val = str(val)
        find = str.find
        for c in val:
            out = out * keylen + find(keyspace, c)
        # dechaffify the value
        out = out // chaffify
        return int(out)