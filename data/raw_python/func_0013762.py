def accept_hotp(key, response, counter, format='dec6', hash=hashlib.sha1,
        drift=3, backward_drift=0):
    '''
       Validate a HOTP value inside a window of
       [counter-backward_drift:counter+forward_drift]

       :param key:
           the shared secret
       :type key:
           hexadecimal string of even length
       :param response:
           the OTP to check
       :type response:
           ASCII string
       :param counter:
           value of the counter running inside an HOTP token, usually it is
           just the count of HOTP value accepted so far for a given shared
           secret; see the specifications of HOTP for more details;
       :param format:
           the output format, can be:
             - hex40, for a 40 characters hexadecimal format,
             - dec4, for a 4 characters decimal format,
             - dec6,
             - dec7, or
             - dec8
           it defaults to dec6.
       :param hash:
           the hash module (usually from the hashlib package) to use,
           it defaults to hashlib.sha1.
       :param drift:
           how far we can look forward from the current value of the counter
       :param backward_drift:
           how far we can look backward from the current counter value to
           match the response, default to zero as it is usually a bad idea to
           look backward as the counter is only advanced when a valid value is
           checked (and so the counter on the token side should have been
           incremented too)

       :returns:
           a pair of a boolean and an integer:
            - first is True if the response is validated and False otherwise,
            - second is the new value for the counter; it can be more than
              counter + 1 if the drift window was used; you must store it if
              the response was validated.

       >>> accept_hotp('343434', '122323', 2, format='dec6')
           (False, 2)

       >>> hotp('343434', 2, format='dec6')
           '791903'

       >>> accept_hotp('343434', '791903', 2, format='dec6')
           (True, 3)

       >>> hotp('343434', 3, format='dec6')
           '907279'

       >>> accept_hotp('343434', '907279', 2, format='dec6')
           (True, 4)
    '''

    for i in range(-backward_drift, drift+1):
        if _utils.compare_digest(hotp(key, counter+i, format=format, hash=hash), str(response)):
            return True, counter+i+1
    return False,counter