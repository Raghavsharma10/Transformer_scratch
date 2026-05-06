def convert_verse_group_elements(self):
        """
        A song, poem, or verse

        Implementor’s Note: No attempt has been made to retain the look or
        visual form of the original poetry.

        This unusual element, <verse-group> is used to convey poetry and is
        recursive in nature (it may contain further <verse-group> elements).
        Examples of these tags are sparse, so it remains difficult to ensure
        full implementation. This method will attempt to handle the label,
        title, and subtitle elements correctly, while converting <verse-lines>
        to italicized lines.
        """
        for verse_group in self.main.getroot().findall('.//verse-group'):
            #Find some possible sub elements for the heading
            label = verse_group.find('label')
            title = verse_group.find('title')
            subtitle = verse_group.find('subtitle')
            #Modify the verse-group element
            verse_group.tag = 'div'
            verse_group.attrib['id'] = 'verse-group'
            #Create a title for the verse_group
            if label is not None or title is not None or subtitle is not None:
                new_verse_title = etree.Element('b')
                #Insert it at the beginning
                verse_group.insert(0, new_verse_title)
                #Induct the title elements into the new title
                if label is not None:
                    append_all_below(new_verse_title, label)
                    remove(label)
                if title is not None:
                    append_all_below(new_verse_title, title)
                    remove(title)
                if subtitle is not None:
                    append_all_below(new_verse_title, subtitle)
                    remove(subtitle)
            for verse_line in verse_group.findall('verse-line'):
                verse_line.tag = 'p'
                verse_line.attrib['class'] = 'verse-line'