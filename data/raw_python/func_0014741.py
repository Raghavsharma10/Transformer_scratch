def addStartTag(contents, startTag):
    '''
        addStartTag - Safetly add a start tag to the document, taking into account the DOCTYPE

        @param contents <str> - Contents
        @param startTag <str> - Fully formed tag, i.e. <html>
    '''

    matchObj = DOCTYPE_MATCH.match(contents)
    if matchObj:
        idx = matchObj.end()
    else:
        idx = 0
    return "%s\n%s\n%s" %(contents[:idx], startTag, contents[idx:])