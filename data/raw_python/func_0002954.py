def make_heading_affiliations(self, heading_div):
        """
        Makes the content for the Author Affiliations, displays after the
        Authors segment in the Heading.

        Metadata element, content derived from FrontMatter
        """
        #Get all of the aff element tuples from the metadata
        affs = self.article.root.xpath('./front/article-meta/aff')
        #Create a list of all those pertaining to the authors
        author_affs = [i for i in affs if 'aff' in i.attrib['id']]
        #Count them, used for formatting
        if len(author_affs) == 0:
            return None
        else:
            affs_list = etree.SubElement(heading_div,
                                         'ul',
                                         {'id': 'affiliations',
                                          'class': 'simple'})

        for aff in author_affs:
            #Create a span element to accept extracted content
            aff_item = etree.SubElement(affs_list, 'li')
            aff_item.attrib['id'] = aff.attrib['id']
            #Get the first label node and the first addr-line node
            label = aff.find('label')
            addr_line = aff.find('addr-line')
            if label is not None:
                bold = etree.SubElement(aff_item, 'b')
                bold.text = all_text(label) + ' '
            if addr_line is not None:
                append_new_text(aff_item, all_text(addr_line))
            else:
                append_new_text(aff_item, all_text(aff))