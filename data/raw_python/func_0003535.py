def quote(myitem, elt=True):
    '''URL encode string'''
    if elt and '<' in myitem and len(myitem) > 24 and myitem.find(']]>') == -1:
        return '<![CDATA[%s]]>' % (myitem)
    else:
        myitem = myitem.replace('&', '&amp;').\
            replace('<', '&lt;').replace(']]>', ']]&gt;')
    if not elt:
        myitem = myitem.replace('"', '&quot;')
    return myitem