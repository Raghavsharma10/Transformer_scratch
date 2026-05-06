def convert_list_elements(self):
        """
        A sequence of two or more items, which may or may not be ordered.

        The <list> element has an optional <label> element and optional <title>
        element, followed by one or more <list-item> elements. This is element
        is recursive as the <list-item> elements may contain further <list> or
        <def-list> elements. Much of the potential complexity in dealing with
        lists comes from this recursion.
        """
        #I have yet to gather many examples of this element, and may have to
        #write a recursive method for the processing of lists depending on how
        #PLoS produces their XML, for now this method is ignorant of nesting

        #TODO: prefix-words, one possible solution would be to have this method
        #edit the CSS to provide formatting support for arbitrary prefixes...

        #This is a block level element, so elevate it if found in p
        for list_el in self.main.getroot().findall('.//list'):
            if list_el.getparent().tag == 'p':
                elevate_element(list_el)

        #list_el is used instead of list (list is reserved)
        for list_el in self.main.getroot().findall('.//list'):
            if 'list-type' not in list_el.attrib:
                list_el_type = 'order'
            else:
                list_el_type = list_el.attrib['list-type']
            #Unordered lists
            if list_el_type in ['', 'bullet', 'simple']:
                list_el.tag = 'ul'
                #CSS must be used to recognize the class and suppress bullets
                if list_el_type == 'simple':
                    list_el.attrib['class'] = 'simple'
            #Ordered lists
            else:
                list_el.tag = 'ol'
                list_el.attrib['class'] = list_el_type
            #Convert the list-item element tags to 'li'
            for list_item in list_el.findall('list-item'):
                list_item.tag = 'li'
            remove_all_attributes(list_el, exclude=['id', 'class'])