def completed_number(prefix, length):
    """
    'prefix' is the start of the CC number as a string, any number of digits.
    'length' is the length of the CC number to generate. Typically 13 or 16
    """
    ccnumber = prefix
    # generate digits
    while len(ccnumber) < (length - 1):
        digit = random.choice(['0',  '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        ccnumber.append(digit)
    # Calculate sum 
    sum = 0
    pos = 0
    reversedCCnumber = []
    reversedCCnumber.extend(ccnumber)
    reversedCCnumber.reverse()
    while pos < length - 1:
        odd = int( reversedCCnumber[pos] ) * 2
        if odd > 9:
            odd -= 9
        sum += odd
        if pos != (length - 2):
            sum += int( reversedCCnumber[pos+1] )
        pos += 2
    # Calculate check digit
    checkdigit = ((sum / 10 + 1) * 10 - sum) % 10
    ccnumber.append( str(int(checkdigit)) )
    return ''.join(ccnumber)