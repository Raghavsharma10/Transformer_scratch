def convert_table_wrap_elements(self):
        """
        Responsible for the correct conversion of JPTS 3.0 <table-wrap>
        elements to EPUB content.

        The 'id' attribute is treated as mandatory by this method.
        """
        for table_wrap in self.main.getroot().findall('.//table-wrap'):

            table_div = etree.Element('div', {'id': table_wrap.attrib['id']})

            label = table_wrap.find('label')
            caption = table_wrap.find('caption')
            alternatives = table_wrap.find('alternatives')
            graphic = table_wrap.find('graphic')
            table = table_wrap.find('table')
            if graphic is None:
                if alternatives is not None:
                    graphic = alternatives.find('graphic')
            if table is None:
                if alternatives is not None:
                    table = alternatives.find('table')

            #Handling the label and caption
            if label is not None and caption is not None:
                caption_div = etree.Element('div', {'class': 'table-caption'})
                caption_div_b = etree.SubElement(caption_div, 'b')
                if label is not None:
                    append_all_below(caption_div_b, label)
                if caption is not None:
                    #Find, optional, title element and paragraph elements
                    caption_title = caption.find('title')
                    if caption_title is not None:
                        append_all_below(caption_div_b, caption_title)
                    caption_ps = caption.findall('p')
                    #For title and each paragraph, give children to the div
                    for caption_p in caption_ps:
                        append_all_below(caption_div, caption_p)
                #Add this to the table div
                table_div.append(caption_div)

            ### Practical Description ###
            #A table may have both, one of, or neither of graphic and table
            #The different combinations should be handled, but a table-wrap
            #with neither should fail with an error
            #
            #If there is both an image and a table, the image should be placed
            #in the text flow with a link to the html table
            #
            #If there is an image and no table, the image should be placed in
            #the text flow without a link to an html table
            #
            #If there is a table with no image, then the table should be placed
            #in the text flow.

            if graphic is not None:
                #Create the image path for the graphic
                xlink_href = ns_format(graphic, 'xlink:href')
                graphic_xlink_href = graphic.attrib[xlink_href]
                file_name = graphic_xlink_href.split('.')[-1] + '.png'
                img_dir = 'images-' + self.doi_suffix()
                img_path = '/'.join([img_dir, file_name])
                #Create the new img element
                img_element = etree.Element('img', {'alt': 'A Table',
                                                    'src': img_path,
                                                    'class': 'table'})
                #Add this to the table div
                table_div.append(img_element)
                #If table, add it to the list, and link to it
                if table is not None:  # Both graphic and table
                    #The label attribute is just a means of transmitting some
                    #plaintext which will be used for the labeling in the html
                    #tables file
                    div = etree.SubElement(self.tables.find('body'),
                                           'div',
                                           {'id': table_wrap.attrib['id']})

                    if label is not None:
                        bold_label = etree.SubElement(div, 'b')
                        append_all_below(bold_label, label)
                    #Add the table to the tables list
                    div.append(deepcopy(table))
                    #Also add the table's foot if it exists
                    table_wrap_foot = table_wrap.find('table-wrap-foot')
                    if table_wrap_foot is not None:
                        table_wrap_foot.tag = 'div'
                        table_wrap_foot.attrib['class'] = 'table-wrap-foot'
                        div.append(table_wrap_foot)
                    #Create a link to the html version of the table
                    html_table_link = etree.Element('a')
                    html_table_link.attrib['href'] = self.tables_fragment.format(table_wrap.attrib['id'])
                    html_table_link.text = 'Go to HTML version of this table'
                    #Add this to the table div
                    table_div.append(html_table_link)
                    remove(table)

            elif table is not None:  # Table only
                #Simply append the table to the table div
                table_div.append(table)
            elif graphic is None and table is None:
                sys.exit('Encountered table-wrap element with neither graphic nor table. Exiting.')

            #Replace the original table-wrap with the newly constructed div
            replace(table_wrap, table_div)