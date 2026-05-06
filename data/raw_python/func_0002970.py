def convert_disp_formula_elements(self):
        """
        <disp-formula> elements must be converted to conforming elements
        """
        for disp in self.main.getroot().findall('.//disp-formula'):
            #find label element
            label_el = disp.find('label')
            graphic_el = disp.find('graphic')
            if graphic_el is None:  # No graphic, assume math as text instead
                text_span = etree.Element('span', {'class': 'disp-formula'})
                if 'id' in disp.attrib:
                    text_span.attrib['id'] = disp.attrib['id']
                append_all_below(text_span, disp)
                #Insert the text span before the disp-formula
                insert_before(disp, text_span)
                #If a label exists, modify and insert before text_span
                if label_el is not None:
                    label_el.tag = 'b'
                    insert_before(text_span, label_el)
                #Remove the disp-formula
                remove(disp)
                #Skip the rest, which deals with the graphic element
                continue
            #The graphic element is present
            #Create a file reference for the image
            xlink_href = ns_format(graphic_el, 'xlink:href')
            graphic_xlink_href = graphic_el.attrib[xlink_href]
            file_name = graphic_xlink_href.split('.')[-1] + '.png'
            img_dir = 'images-' + self.doi_suffix()
            img_path = '/'.join([img_dir, file_name])

            #Create the img element
            img_element = etree.Element('img', {'alt': 'A Display Formula',
                                                'class': 'disp-formula',
                                                'src': img_path})
            #Transfer the id attribute
            if 'id' in disp.attrib:
                img_element.attrib['id'] = disp.attrib['id']
            #Insert the img element
            insert_before(disp, img_element)
            #Create content for the label
            if label_el is not None:
                label_el.tag = 'b'
                insert_before(img_element, label_el)
            #Remove the old disp-formula element
            remove(disp)