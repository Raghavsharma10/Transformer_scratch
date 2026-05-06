def str2cryptofunction(crypto_function_description):
    '''
       Convert an OCRA crypto function description into a CryptoFunction
       instance

       :param crypto_function_description:
       :returns:
           the CryptoFunction object
       :rtype: CryptoFunction
    '''
    s = crypto_function_description.split('-')
    if len(s) != 3:
        raise ValueError('CryptoFunction description must be triplet separated by -')
    if s[0] != HOTP:
        raise ValueError('Unknown CryptoFunction kind %s' % s[0])
    algo = str2hashalgo(s[1])
    try:
        truncation_length = int(s[2])
        if truncation_length < 0 or truncation_length > 10:
            raise ValueError()
    except ValueError:
        raise ValueError('Invalid truncation length %s' % s[2])
    return CryptoFunction(algo, truncation_length)