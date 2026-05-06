def minify(text, minifier):
    '''Minifies the source text (if needed)'''
    # there really isn't a good way to know if a file is already minified.
    # our heuristic is if source is more than 50 bytes greater of dest OR
    # if a hard return is found in the first 50 chars, we assume it is not minified.
    minified = minifier(text)
    if  abs(len(text) - len(minified)) > 50 or '\n' in text[:50]:
        return minified
    return text