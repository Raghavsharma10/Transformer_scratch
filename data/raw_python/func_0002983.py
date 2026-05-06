def convert_graphic_elements(self):
        """
        This is a method for the odd special cases where <graphic> elements are
        standalone, or rather, not a part of a standard graphical element such
        as a figure or a table. This method should always be employed after the
        standard cases have already been handled.
        """
        for graphic in self.main.getroot().findall('.//graphic'):
            graphic.tag = 'img'
            graphic.attrib['alt'] = 'unowned-graphic'
            ns_xlink_href = ns_format(graphic, 'xlink:href')
            if ns_xlink_href in graphic.attrib:
                xlink_href = graphic.attrib[ns_xlink_href]
                file_name = xlink_href.split('.')[-1] + '.png'
                img_dir = 'images-' + self.doi_suffix()
                img_path = '/'.join([img_dir, file_name])
                graphic.attrib['src'] = img_path
            remove_all_attributes(graphic, exclude=['id', 'class', 'alt', 'src'])