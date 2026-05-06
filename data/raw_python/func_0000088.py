def extract_transformers_from_source(source):
    '''Scan a source for lines of the form
       from __experimental__ import transformer1 [,...]
       identifying transformers to be used. Such line is passed to the
       add_transformer function, after which it is removed from the
       code to be executed.
    '''
    lines = source.split('\n')
    linenumbers = []
    for number, line in enumerate(lines):
        if FROM_EXPERIMENTAL.match(line):
            add_transformers(line)
            linenumbers.insert(0, number)

    # drop the "fake" import from the source code
    for number in linenumbers:
        del lines[number]
    return '\n'.join(lines)