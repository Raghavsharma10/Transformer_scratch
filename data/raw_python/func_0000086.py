def add_transformers(line):
    '''Extract the transformers names from a line of code of the form
       from __experimental__ import transformer1 [,...]
       and adds them to the globally known dict
    '''
    assert FROM_EXPERIMENTAL.match(line)

    line = FROM_EXPERIMENTAL.sub(' ', line)
    # we now have: " transformer1 [,...]"
    line = line.split("#")[0]    # remove any end of line comments
    # and insert each transformer as an item in a list
    for trans in line.replace(' ', '').split(','):
        import_transformer(trans)