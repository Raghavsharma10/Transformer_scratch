def find_common_beginning(string_list, boundary_char = None):
    """Given a list of strings, finds finds the longest string that is common
    to the *beginning* of all strings in the list.
    
    boundary_char defines a boundary that must be preserved, so that the
    common string removed must end with this char.
    """
    
    common=''
    
    # by definition there is nothing common to 1 item...
    if len(string_list) > 1:
        shortestLen = min([len(el) for el in string_list])
        
        for idx in range(shortestLen):
            chars = [s[idx] for s in string_list]
            if chars.count(chars[0]) != len(chars): # test if any chars differ
                break
            common+=chars[0]
    
        
    if boundary_char is not None:
        try:
            end_idx = common.rindex(boundary_char)
            common = common[0:end_idx+1]
        except ValueError:
            common = ''
    
    return common