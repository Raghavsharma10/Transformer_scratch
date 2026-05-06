def escapejson(string):
    '''
    Escape `string`, which should be syntactically valid JSON (this is not
    verified), so that it is safe for inclusion in HTML <script> environments
    and as literal javascript.
    '''
    for fro, to in replacements:
        string = string.replace(fro, to)
    return string