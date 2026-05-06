def convert_boxed_text_elements(self):
        """
        Textual material that is part of the body of text but outside the
        flow of the narrative text, for example, a sidebar, marginalia, text
        insert (whether enclosed in a box or not), caution, tip, note box, etc.

        <boxed-text> elements for PLoS appear to all contain a single <sec>
        element which frequently contains a <title> and various other content.
        This method will elevate the <sec> element, adding class information as
        well as processing the title.
        """
        for boxed_text in self.main.getroot().findall('.//boxed-text'):
            sec_el = boxed_text.find('sec')
            if sec_el is not None:
                sec_el.tag = 'div'
                title = sec_el.find('title')
                if title is not None:
                    title.tag = 'b'
                sec_el.attrib['class'] = 'boxed-text'
                if 'id' in boxed_text.attrib:
                    sec_el.attrib['id'] = boxed_text.attrib['id']
                replace(boxed_text, sec_el)
            else:
                div_el = etree.Element('div', {'class': 'boxed-text'})
                if 'id' in boxed_text.attrib:
                    div_el.attrib['id'] = boxed_text.attrib['id']
                append_all_below(div_el, boxed_text)
                replace(boxed_text, div_el)