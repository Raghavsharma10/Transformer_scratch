def make_heading_abstracts(self, heading_div):
        """
        An article may contain data for various kinds of abstracts. This method
        works on those that are included in the Heading. This is displayed
        after the Authors and Affiliations.

        Metadata element, content derived from FrontMatter
        """
        for abstract in self.article.root.xpath('./front/article-meta/abstract'):
            #Make a copy of the abstract
            abstract_copy = deepcopy(abstract)
            abstract_copy.tag = 'div'
            #Abstracts are a rather diverse bunch, keep an eye on them!
            title_text = abstract_copy.xpath('./title[1]/text()')
            for title in abstract_copy.findall('.//title'):
                remove(title)
            #Create a header for the abstract
            abstract_header = etree.Element('h2')
            remove_all_attributes(abstract_copy)
            #Set the header text and abstract id according to abstract type
            abstract_type = abstract.attrib.get('abstract-type')
            log.debug('Handling Abstrace of with abstract-type="{0}"'.format(abstract_type))
            if abstract_type == 'summary':
                abstract_header.text = 'Author Summary'
                abstract_copy.attrib['id'] = 'author-summary'
            elif abstract_type == 'editors-summary':
                abstract_header.text = 'Editors\' Summary'
                abstract_copy.attrib['id'] = 'editor-summary'
            elif abstract_type == 'synopsis':
                abstract_header.text = 'Synopsis'
                abstract_copy.attrib['id'] = 'synopsis'
            elif abstract_type == 'alternate':
                #Right now, these will only be included if there is a title to
                #give it
                if title_text:
                    abstract_header.text= title_text[0]
                    abstract_copy.attrib['id'] = 'alternate'
                else:
                    continue
            elif abstract_type is None:
                abstract_header.text = 'Abstract'
                abstract_copy.attrib['id'] = 'abstract'
            elif abstract_type == 'toc':  # We don't include these
                continue
            else:  # Warn about these, then skip
                log.warning('No handling for abstract-type="{0}"'.format(abstract_type))
                continue
                #abstract_header.text = abstract_type
                #abstract_copy.attrib['id'] = abstract_type
            heading_div.append(abstract_header)
            heading_div.append(abstract_copy)