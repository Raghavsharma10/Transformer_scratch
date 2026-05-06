def convert_fn_elements(self):
        """
        <fn> elements may be used in the main text body outside of tables and
        figures for purposes such as erratum notes. It appears that PLoS
        practice is to not show erratum notes in the web or pdf formats after
        the appropriate corrections have been made to the text. The erratum
        notes are thus the only record that an error was made.

        This method will attempt to display footnotes unless the note can be
        identified as an Erratum, in which case it will be removed in
        accordance with PLoS' apparent guidelines.
        """
        for footnote in self.main.getroot().findall('.//fn'):
            #Use only the first paragraph
            paragraph = footnote.find('p')
            #If no paragraph, move on
            if paragraph is None:
                remove(footnote)
                continue
            #Simply remove corrected errata items
            paragraph_text = str(etree.tostring(paragraph, method='text', encoding='utf-8'), encoding='utf-8')
            if paragraph_text.startswith('Erratum') and 'Corrected' in paragraph_text:
                remove(footnote)
                continue
            #Transfer some attribute information from the fn element to the paragraph
            if 'id' in footnote.attrib:
                paragraph.attrib['id'] = footnote.attrib['id']
            if 'fn-type' in footnote.attrib:
                paragraph.attrib['class'] = 'fn-type-{0}'.footnote.attrib['fn-type']
            else:
                paragraph.attrib['class'] = 'fn'
                #Replace the
            replace(footnote, paragraph)