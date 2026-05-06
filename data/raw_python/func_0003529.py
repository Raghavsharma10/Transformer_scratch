def _transstat(status, grouppath, dictpath, line):
    """Executes processing steps when reading a line"""
    if status == 0:
        raise MTLParseError(
            "Status should not be '%s' after reading line:\n%s"
            % (STATUSCODE[status], line))
    elif status == 1:
        currentdict = dictpath[-1]
        currentgroup = _getgroupname(line)
        grouppath.append(currentgroup)
        currentdict[currentgroup] = {}
        dictpath.append(currentdict[currentgroup])
    elif status == 2:
        currentdict = dictpath[-1]
        newkey, newval = _getmetadataitem(line)

        # USGS has started quoting the scene center time.  If this
        # happens strip quotes before post processing.
        if newkey == 'SCENE_CENTER_TIME' and newval.startswith('"') \
                and newval.endswith('"'):
            # logging.warning('Strip quotes off SCENE_CENTER_TIME.')
            newval = newval[1:-1]

        currentdict[newkey] = _postprocess(newval)
    elif status == 3:
        oldgroup = _getendgroupname(line)
        if oldgroup != grouppath[-1]:
            raise MTLParseError(
                "Reached line '%s' while reading group '%s'."
                % (line.strip(), grouppath[-1]))
        del grouppath[-1]
        del dictpath[-1]
        try:
            currentgroup = grouppath[-1]
        except IndexError:
            currentgroup = None
    elif status == 4:
        if grouppath:
            raise MTLParseError(
                "Reached end before end of group '%s'" % grouppath[-1])
    return grouppath, dictpath