def convert_fig_elements(self):
        """
        Responsible for the correct conversion of JPTS 3.0 <fig> elements to
        EPUB xhtml. Aside from translating <fig> to <img>, the content model
        must be edited.
        """
        for fig in self.main.getroot().findall('.//fig'):
            if fig.getparent().tag == 'p':
                elevate_element(fig)
        for fig in self.main.getroot().findall('.//fig'):
            #self.convert_fn_elements(fig)
            #self.convert_disp_formula_elements(fig)
            #Find label and caption
            label_el = fig.find('label')
            caption_el = fig.find('caption')
            #Get the graphic node, this should be mandatory later on
            graphic_el = fig.find('graphic')
            #Create a file reference for the image
            xlink_href = ns_format(graphic_el, 'xlink:href')
            graphic_xlink_href = graphic_el.attrib[xlink_href]
            file_name = graphic_xlink_href.split('.')[-1] + '.png'
            img_dir = 'images-' + self.doi_suffix()
            img_path = '/'.join([img_dir, file_name])

            #Create the content: using image path, label, and caption
            img_el = etree.Element('img', {'alt': 'A Figure', 'src': img_path,
                                           'class': 'figure'})
            if 'id' in fig.attrib:
                img_el.attrib['id'] = fig.attrib['id']
            insert_before(fig, img_el)

            #Create content for the label and caption
            if caption_el is not None or label_el is not None:
                img_caption_div = etree.Element('div', {'class': 'figure-caption'})
                img_caption_div_b = etree.SubElement(img_caption_div, 'b')
                if label_el is not None:
                    append_all_below(img_caption_div_b, label_el)
                    append_new_text(img_caption_div_b, '. ', join_str='')
                if caption_el is not None:
                    caption_title = caption_el.find('title')
                    if caption_title is not None:
                        append_all_below(img_caption_div_b, caption_title)
                        append_new_text(img_caption_div_b, ' ', join_str='')
                    for each_p in caption_el.findall('p'):
                        append_all_below(img_caption_div, each_p)
                insert_before(fig, img_caption_div)

            #Remove the original <fig>
            remove(fig)