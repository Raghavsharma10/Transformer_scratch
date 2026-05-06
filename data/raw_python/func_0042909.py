def readScriptRanges(scripts=None):
    """
    Read script ranges from http://unicode.org/Public/UNIDATA/Scripts.txt file.
    """
    scripts = scripts or LATIN_LIKE_SCRIPTS
    ranges = []

    f = open('Scripts.txt', 'r')
    for line in f:
        line = line.strip('\n')
        matchObj = re.match(
            '^([0123456789ABCDEF]{4}(\.\.[0123456789ABCDEF]{4})?)\s*;\s+(%s)\s+(#.*)?$'
                % '|'.join(scripts),
            line)
        if matchObj:
            entry = matchObj.group(1)
            if len(entry) > 4:
                start, stop = entry.split('..', 1)
                ranges.append((start, stop))
            else:
                ranges.append(entry)
    f.close()

    return ranges