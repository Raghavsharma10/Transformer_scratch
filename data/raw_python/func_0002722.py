def render_EPUB2(self, location):
        """
        Creates the NCX specified file for EPUB2
        """

        def make_navlabel(text):
            """
            Creates and returns a navLabel element with the supplied text.
            """
            navlabel = etree.Element('navLabel')
            navlabel_text = etree.SubElement(navlabel, 'text')
            navlabel_text.text = text
            return navlabel

        def make_navMap(nav=None):
            if nav is None:
                nav_element = etree.Element('navMap')
                for nav_point in self.nav:
                    nav_element.append(make_navMap(nav=nav_point))
            else:
                nav_element = etree.Element('navPoint')
                nav_element.attrib['id'] = nav.id
                nav_element.attrib['playOrder'] = nav.playOrder
                nav_element.append(make_navlabel(nav.label))
                content_element = etree.SubElement(nav_element, 'content')
                content_element.attrib['src'] = nav.source
                for child in nav.children:
                    nav_element.append(make_navMap(nav=child))
            return nav_element
        root = etree.XML('''\
<?xml version="1.0"?>\
<ncx version="2005-1" xmlns="http://www.daisy.org/z3986/2005/ncx/">\
<head>\
<meta name="dtb:uid" content="{uid}"/>\
<meta name="dtb:depth" content="{depth}"/>\
<meta name="dtb:totalPageCount" content="0"/>\
<meta name="dtb:maxPageNumber" content="0"/>\
<meta name="dtb:generator" content="OpenAccess_EPUB {version}"/>\
</head>\
</ncx>'''.format(**{'uid': ','.join(self.all_dois),
                    'depth': self.nav_depth,
                    'version': __version__}))
        document = etree.ElementTree(root)
        ncx = document.getroot()

        #Create the docTitle element
        doctitle = etree.SubElement(ncx, 'docTitle')
        doctitle_text = etree.SubElement(doctitle, 'text')
        doctitle_text.text = self.title

        #Create the docAuthor elements
        for contributor in self.contributors:
            if contributor.role == 'author':
                docauthor = etree.SubElement(ncx, 'docAuthor')
                docauthor_text = etree.SubElement(docauthor, 'text')
                docauthor_text.text = contributor.name

        #Create the navMap element
        ncx.append(make_navMap())

        if self.figures_list:
            navlist = etree.SubElement(ncx, 'navList')
            navlist.append(make_navlabel('List of Figures'))
            for nav_pt in self.figures_list:
                navtarget = etree.SubElement(navlist, 'navTarget')
                navtarget.attrib['id'] = nav_pt.id
                navtarget.append(self.make_navlabel(nav_pt.label))
                content = etree.SubElement(navtarget, 'content')
                content.attrib['src'] = nav_pt.source

        if self.tables_list:
            navlist = etree.SubElement(ncx, 'navList')
            navlist.append(make_navlabel('List of Tables'))
            for nav_pt in self.tables_list:
                navtarget = etree.SubElement(navlist, 'navTarget')
                navtarget.attrib['id'] = nav_pt.id
                navtarget.append(self.make_navlabel(nav_pt.label))
                content = etree.SubElement(navtarget, 'content')
                content.attrib['src'] = nav_pt.source

        with open(os.path.join(location, 'EPUB', 'toc.ncx'), 'wb') as output:
            output.write(etree.tostring(document, encoding='utf-8', pretty_print=True))