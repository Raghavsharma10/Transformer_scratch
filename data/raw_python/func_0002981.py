def convert_ref_list_elements(self):
        """
        List of references (citations) for an article, which is often called
        “References”, “Bibliography”, or “Additional Reading”.

        No distinction is made between lists of cited references and lists of
        suggested references.

        This method should not be confused with the method(s) employed for the
        formatting of a proper bibliography, though they are related.
        Similarly, this is an area of major openness in development, I lack
        access to PLOS' algorithm for proper citation formatting.
        """
        #TODO: Handle nested ref-lists
        for ref_list in self.main.getroot().findall('.//ref-list'):
            remove_all_attributes(ref_list)
            ref_list.tag = 'div'
            ref_list.attrib['class'] = 'ref-list'
            label = ref_list.find('label')
            if label is not None:
                label.tag = 'h3'
            for ref in ref_list.findall('ref'):
                ref_p = etree.Element('p')
                ref_p.text = str(etree.tostring(ref, method='text', encoding='utf-8'), encoding='utf-8')
                replace(ref, ref_p)