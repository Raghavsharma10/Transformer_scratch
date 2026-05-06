def flattened2str(flattened, missing=False, extra=False):
    """ Return a pretty-printed multi-line string version of the output of
    flatten_errors. Know that flattened comes in the form of a list
    of keys that failed. Each member of the list is a tuple::

        ([list of sections...], key, result)

    so we turn that into a string. Set missing to True if all the input
    problems are from missing items.  Set extra to True if all the input
    problems are from extra items. """

    if flattened is None or len(flattened) < 1:
        return ''
    retval = ''
    for sections, key, result in flattened:
        # Name the section and item, to start the message line
        if sections is None or len(sections) == 0:
            retval += '\t"'+key+'"'
        elif len(sections) == 1:
            if key is None:
                # a whole section is missing at the top-level; see if hidden
                junk = sections[0]
                if isHiddenName(junk):
                    continue # this missing or extra section is not an error
                else:
                    retval += '\tSection "'+sections[0]+'"'
            else:
                retval += '\t"'+sections[0]+'.'+key+'"'
        else: # len > 1
            joined = '.'.join(sections)
            joined = '"'+joined+'"'
            if key is None:
                retval +=  '\tSection '+joined
            else:
                retval +=  '\t"'+key+'" from '+joined
        # End the msg line with "what seems to be the trouble" with this one
        if missing and result==False:
            retval += ' is missing.'
        elif extra:
            if result:
                retval += ' is an unexpected section. Is your file out of date?'
            else:
                retval += ' is an unexpected parameter. Is your file out of date?'
        elif isinstance(result, bool):
            retval += ' has an invalid value'
        else:
            retval += ' is invalid, '+result.message
        retval += '\n\n'
    return retval.rstrip()