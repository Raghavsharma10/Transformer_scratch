def convert_html_to_xml(self):
        """
        Parses the HTML parsed texts and converts its tags to XML valid tags.

        :returns: HTML enabled text in a XML valid format.
        :rtype: str
        """

        if hasattr(self, 'content') and self.content != '':
            regex = r'<(?!/)(?!!)'
            xml_content = re.sub(regex, '<xhtml:', self.content)
            return xml_content
        else:
            return ''