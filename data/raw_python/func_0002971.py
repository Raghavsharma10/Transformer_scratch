def convert_inline_formula_elements(self):
        """
        <inline-formula> elements must be converted to be conforming

        These elements may contain <inline-graphic> elements, textual content,
        or both.
        """
        for inline in self.main.getroot().findall('.//inline-formula'):
            #inline-formula elements will be modified in situ
            remove_all_attributes(inline)
            inline.tag = 'span'
            inline.attrib['class'] = 'inline-formula'
            inline_graphic = inline.find('inline-graphic')
            if inline_graphic is None:
                # Do nothing more if there is no graphic
                continue
            #Need to conver the inline-graphic element to an img element
            inline_graphic.tag = 'img'
            #Get a copy of the attributes, then remove them
            inline_graphic_attributes = copy(inline_graphic.attrib)
            remove_all_attributes(inline_graphic)
            #Create a file reference for the image
            xlink_href = ns_format(inline_graphic, 'xlink:href')
            graphic_xlink_href = inline_graphic_attributes[xlink_href]
            file_name = graphic_xlink_href.split('.')[-1] + '.png'
            img_dir = 'images-' + self.doi_suffix()
            img_path = '/'.join([img_dir, file_name])
            #Set the source to the image path
            inline_graphic.attrib['src'] = img_path
            inline_graphic.attrib['class'] = 'inline-formula'
            inline_graphic.attrib['alt'] = 'An Inline Formula'