def listwrap(value):
    """
    PERFORMS THE FOLLOWING TRANSLATION
    None -> []
    value -> [value]
    [...] -> [...]  (unchanged list)

    ##MOTIVATION##
    OFTEN IT IS NICE TO ALLOW FUNCTION PARAMETERS TO BE ASSIGNED A VALUE,
    OR A list-OF-VALUES, OR NULL.  CHECKING FOR WHICH THE CALLER USED IS
    TEDIOUS.  INSTEAD WE CAST FROM THOSE THREE CASES TO THE SINGLE CASE
    OF A LIST

    # BEFORE
    def do_it(a):
        if a is None:
            return
        if not isinstance(a, list):
            a=[a]
        for x in a:
            # do something

    # AFTER
    def do_it(a):
        for x in listwrap(a):
            # do something

    """
    if value == None:
        return FlatList()
    elif is_list(value):
        return wrap(value)
    elif isinstance(value, set):
        return wrap(list(value))
    else:
        return wrap([unwrap(value)])