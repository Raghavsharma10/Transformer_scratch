def _discover_toc(zf, opf_xmldoc, opf_filepath):
    '''
    Returns a list of objects: {title: str, src: str, level: int, index: int}
    '''
    toc = None

    # ePub 3.x
    tag = find_tag(opf_xmldoc, 'item', 'properties', 'nav')
    if tag and 'href' in tag.attributes.keys():
        filepath = unquote(tag.attributes['href'].value)
        # The xhtml file path is relative to the OPF file
        base_dir = os.path.dirname(opf_filepath)
        # print('- Reading Nav file: {}/{}'.format(base_dir, filepath))
        npath = os.path.normpath(os.path.join(base_dir, filepath))
        nav_content = zf.read(npath)
        toc_xmldoc = minidom.parseString(nav_content)

        _toc = []

        for n in toc_xmldoc.getElementsByTagName('a'):
            if n.firstChild and ('href' in n.attributes.keys()):
                href = unquote(n.attributes['href'].value)
                # Discarding CFI links
                if '.html' in href or '.xhtml' in href:
                    title = n.firstChild.nodeValue
                    # try the second node too (maybe the first child is an empty span)
                    if not title and n.firstChild.firstChild:
                        title = n.firstChild.firstChild.nodeValue

                    title = title.strip() if title else None

                    if title:
                        level = -1
                        parentNode = n.parentNode
                        avoid_infinite_loop = 0 # simple security issue to avoid infinite loop for bad epub files
                        while parentNode and parentNode.nodeName != 'nav' and avoid_infinite_loop < 50:
                            if parentNode.nodeName == 'ol': # count the depth of the a link related to ol items
                                level += 1
                            parentNode = parentNode.parentNode
                            avoid_infinite_loop += 1
                        level = max(level, 0) # root level is 0, not -1

                        _toc.append({'title': title, 'src': href, 'level': level})
        if _toc:
            toc = _toc

    if not toc:
        # ePub 2.x
        tag = find_tag(opf_xmldoc, 'item', 'id', 'ncx')
        if not tag:
            tag = find_tag(opf_xmldoc, 'item', 'id', 'ncxtoc')
        if tag and 'href' in tag.attributes.keys():
            filepath = unquote(tag.attributes['href'].value)
            # The ncx file path is relative to the OPF file
            base_dir = os.path.dirname(opf_filepath)
            # print('- Reading NCX file: {}/{}'.format(base_dir, filepath))
            npath = os.path.normpath(os.path.join(base_dir, filepath))
            ncx_content = zf.read(npath)

            toc_xmldoc = minidom.parseString(ncx_content)

            def read_nav_point(nav_point_node, level = 0):
                items = []
                item = {'title': None, 'src': None, 'level': level}
                children_points = []
                for item_node in nav_point_node.childNodes:
                    if item_node.nodeName in ('navLabel', 'ncx:navLabel'):
                        try:
                            text = item_node.getElementsByTagName('text')[0].firstChild
                        except IndexError:
                            try:
                                text = item_node.getElementsByTagName('ncx:text')[0].firstChild
                            except IndexError:
                                text = None

                        item['title'] = text.nodeValue.strip() if text and text.nodeValue else None
                    elif item_node.nodeName in ('content', 'ncx:content'):
                        if item_node.hasAttribute('src'):
                            item['src'] = item_node.attributes['src'].value
                    elif item_node.nodeName in ('navPoint', 'ncx:navPoint'):
                        children_points.append(item_node)

                if item['title']:
                    items.append(item)
                    for child_node in children_points:
                        subitems = read_nav_point(child_node, level=level + 1)
                        items.extend(subitems)
                return items

            def read_nav_map(toc_xmldoc, level=0):
                items = []
                try:
                    nav_map_node = toc_xmldoc.getElementsByTagName('navMap')[0]
                except IndexError:
                    # Some ebooks use the ncx: namespace so try that too
                    try:
                        nav_map_node = toc_xmldoc.getElementsByTagName('ncx:navMap')[0]
                    except IndexError:
                        print('Failed reading TOC')
                        return items

                for nav_point in nav_map_node.childNodes:
                    if nav_point.nodeName in ('navPoint', 'ncx:navPoint'):
                        subitems = read_nav_point(nav_point, level=level)
                        items.extend(subitems)
                return items

            toc = read_nav_map(toc_xmldoc)

    # add indexes
    if toc:
        for i, t in enumerate(toc):
            t['index'] = i

    return toc