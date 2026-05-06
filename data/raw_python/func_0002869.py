def elevate_element(node, adopt_name=None, adopt_attrs=None):
        """
        This method serves a specialized function. It comes up most often when
        working with block level elements that may not be contained within
        paragraph elements, which are presented in the source document as
        inline elements (inside a paragraph element).

        It would be inappropriate to merely insert the block element at the
        level of the parent, since this disorders the document by placing
        the child out of place with its siblings. So this method will elevate
        the node to the parent level and also create a new parent to adopt all
        of the siblings after the elevated child.

        The adopting parent node will have identical attributes and tag name
        as the original parent unless specified otherwise.
        """

        #These must be collected before modifying the xml
        parent = node.getparent()
        grandparent = parent.getparent()
        child_index = parent.index(node)
        parent_index = grandparent.index(parent)
        #Get a list of the siblings
        siblings = list(parent)[child_index+1:]
        #Insert the node after the parent
        grandparent.insert(parent_index+1, node)
        #Only create the adoptive parent if there are siblings
        if len(siblings) > 0 or node.tail is not None:
            #Create the adoptive parent
            if adopt_name is None:
                adopt = etree.Element(parent.tag)
            else:
                adopt = etree.Element(adopt_name)
            if adopt_attrs is None:
                for key in parent.attrib.keys():
                    adopt.attrib[key] = parent.attrib[key]
            else:
                for key in adopt_attrs.keys():
                    adopt.attrib[key] = adopt_attrs[key]
            #Insert the adoptive parent after the elevated child
            grandparent.insert(grandparent.index(node)+1, adopt)
        #Transfer the siblings to the adoptive parent
        for sibling in siblings:
            adopt.append(sibling)
        #lxml's element.tail attribute presents a slight problem, requiring the
        #following oddity
        #Set the adoptive parent's text to the node.tail
        if node.tail is not None:
            adopt.text = node.tail
            node.tail = None