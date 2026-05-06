def hyphen_range(string):
    '''
    Expands a string of numbers separated by commas and hyphens into a list of integers.
    For example:  2-3,5-7,20-21,23,100-200
    '''
    list_numbers = list()
    temporary_list = string.split(',')

    for element in temporary_list:
        sub_element = element.split('-')

        if len(sub_element) == 1:
            list_numbers.append(int(sub_element[0]))
        elif len(sub_element) == 2:
            for x in range(int(sub_element[0]), int(sub_element[1])+1):
                list_numbers.append(x)
        else:
            raise Exception('Something went wrong expanding the range'.format(string))

    return list_numbers