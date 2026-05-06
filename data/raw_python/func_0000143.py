def get_unique_variable_names(text, nb):
    '''returns a list of possible variables names that
       are not found in the original text.'''
    base_name = '__VAR_'
    var_names = []
    i = 0
    j = 0
    while j < nb:
        tentative_name = base_name + str(i)
        if text.count(tentative_name) == 0 and tentative_name not in ALL_NAMES:
            var_names.append(tentative_name)
            ALL_NAMES.append(tentative_name)
            j += 1
        i += 1
    return var_names