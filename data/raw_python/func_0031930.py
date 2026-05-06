def _draw_tickgram(numbers):
    """
        Takes a list of integers and generate the equivalent list 
        of ticks corresponding to each of the number
    """
    max_number = max(filter(lambda x : type(x)==int,numbers))
    # If the maxium number is 0, then all the numbers should be 0
    # coz we have called handle_negatives prior to this function
    if max_number == 0 :
        return upticks[0]*len(numbers)
    else:
        normalized_numbers = [ float(x)/max_number if type(x)==int else x for x in numbers ]
        upticks_indexes = [ int(math.ceil(x*len(upticks))) if type(x)==float else x for x in normalized_numbers ]
        return ''.join([ ' ' if type(x)==str else upticks[x-1] if x != 0 else upticks[0] for x in upticks_indexes ])