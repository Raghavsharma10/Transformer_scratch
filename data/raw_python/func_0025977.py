def printCols(strlist,cols=5,width=80):

    """Print elements of list in cols columns"""

    # This may exist somewhere in the Python standard libraries?
    # Should probably rewrite this, it is pretty crude.

    nlines = (len(strlist)+cols-1)//cols
    line = nlines*[""]
    for i in range(len(strlist)):
        c, r = divmod(i,nlines)
        nwid = c*width//cols - len(line[r])
        if nwid>0:
            line[r] = line[r] + nwid*" " + strlist[i]
        else:
            line[r] = line[r] + " " + strlist[i]
    for s in line:
        print(s)