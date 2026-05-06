def convert_supplementary_material_elements(self):
        """
        Supplementary material are not, nor are they generally expected to be,
        packaged into the epub file. Though this is a technical possibility,
        and certain epub reading systems (such as those run on a PC) might be
        reasonably capable of the external handling of diverse file formats
        I presume that supplementary material will remain separate from the
        document. So special cases aside, external links to supplementary
        material will be employed; this will require internet connection for
        access.

        As for content in <supplementary-material>, they appear to strictly
        contain 1 <label> element, followed by a <caption><title><p></caption>
        substructure.
        """
        for supplementary in self.main.getroot().findall('.//supplementary-material'):
            #Create a div element to hold the supplementary content
            suppl_div = etree.Element('div')
            if 'id' in supplementary.attrib:
                suppl_div.attrib['id'] = supplementary.attrib['id']
            insert_before(supplementary, suppl_div)
            #Get the sub elements
            label = supplementary.find('label')
            caption = supplementary.find('caption')
            #Get the external resource URL for the supplementary information
            ns_xlink_href = ns_format(supplementary, 'xlink:href')
            xlink_href = supplementary.attrib[ns_xlink_href]
            resource_url = self.fetch_single_representation(xlink_href)
            if label is not None:
                label.tag = 'a'
                label.attrib['href'] = resource_url
                append_new_text(label, '. ', join_str='')
                suppl_div.append(label)
            if caption is not None:
                title = caption.find('title')
                paragraphs = caption.findall('p')
                if title is not None:
                    title.tag = 'b'
                    suppl_div.append(title)
                for paragraph in paragraphs:
                    suppl_div.append(paragraph)
            #This is a fix for odd articles with <p>s outside of <caption>
            #See journal.pctr.0020006, PLoS themselves fail to format this for
            #the website, though the .pdf is good
            #It should be noted that journal.pctr.0020006 does not pass
            #validation because it places a <p> before a <caption>
            #By placing this at the end of the method, it conforms to the spec
            #by expecting such p tags after caption. This causes a hiccup in
            #the rendering for journal.pctr.0020006, but it's better than
            #skipping the data entirely AND it should also work for conforming
            #articles.
            for paragraph in supplementary.findall('p'):
                suppl_div.append(paragraph)
            remove(supplementary)