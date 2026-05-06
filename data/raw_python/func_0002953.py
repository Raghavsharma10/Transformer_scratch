def make_heading_authors(self, authors):
        """
        Constructs the Authors content for the Heading. This should display
        directly after the Article Title.

        Metadata element, content derived from FrontMatter
        """
        author_element = etree.Element('h3', {'class': 'authors'})
        #Construct content for the author element
        first = True
        for author in authors:
            if first:
                first = False
            else:
                append_new_text(author_element, ',', join_str='')
            collab = author.find('collab')
            anon = author.find('anon')
            if collab is not None:
                append_all_below(author_element, collab)
            elif anon is not None:  # If anonymous, just add "Anonymous"
                append_new_text(author_element, 'Anonymous')
            else:  # Author is neither Anonymous or a Collaboration
                author_name, _ = self.get_contrib_names(author)
                append_new_text(author_element, author_name)
            #TODO: Handle author footnote references, also put footnotes in the ArticleInfo
            #Example: journal.pbio.0040370.xml
            first = True
            for xref in author.xpath("./xref[@ref-type='corresp' or @ref-type='aff']"):
                _sup = xref.find('sup')
                sup_text = all_text(_sup) if _sup is not None else ''
                auth_sup = etree.SubElement(author_element, 'sup')
                sup_link = etree.SubElement(auth_sup,
                                            'a',
                                            {'href': self.main_fragment.format(xref.attrib['rid'])})
                sup_link.text = sup_text
                if first:
                    first = False
                else:
                    append_new_text(auth_sup, ', ', join_str='')
            #for xref in author.findall('xref'):
                #if xref.attrs['ref-type'] in ['corresp', 'aff']:

                    #try:
                        #sup_element = xref.sup[0].node
                    #except IndexError:
                        #sup_text = ''
                    #else:
                        #sup_text = all_text(sup_element)
                    #new_sup = etree.SubElement(author_element, 'sup')
                    #sup_link = etree.SubElement(new_sup, 'a')
                    #sup_link.attrib['href'] = self.main_fragment.format(xref.attrs['rid'])
                    #sup_link.text = sup_text
                    #if first:
                        #first = False
                    #else:
                        #new_sup.text = ','
        return author_element