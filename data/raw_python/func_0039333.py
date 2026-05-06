def numlistbetween(num1, num2, option='list', listoption='string'):
    """
    List Or Count The Numbers Between Two Numbers
    """
    if option == 'list':
        if listoption == 'string':
            output = ''
            output += str(num1)
            for currentnum in range(num1 + 1, num2 + 1):
                output += ','
                output += str(currentnum)
        elif listoption == 'list':
            output = []
            for currentnum in range(num1, num2 + 1):
                output.append(str(currentnum))
            return output
    elif option == 'count':
        return num2 - num1