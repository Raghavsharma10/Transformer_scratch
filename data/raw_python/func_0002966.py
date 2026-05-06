def make_back_acknowledgments(self):
        """
        The <ack> is an important piece of back matter information, and will be
        including immediately after the main text.

        This element should only occur once, optionally, for PLoS, if a need
        becomes known, then multiple instances may be supported.
        """
        acks = self.article.root.xpath('./back/ack')
        if not acks:
            return
        ack = deepcopy(acks[0])
        #Modify the tag to div
        ack.tag = 'div'
        #Give it an id
        ack.attrib['id'] = 'acknowledgments'
        #Give it a title element--this is not an EPUB element but doing so will
        #allow it to later be depth-formatted by self.convert_div_titles()
        ack_title = etree.Element('title')
        ack_title.text = 'Acknowledgments'
        ack.insert(0, ack_title)  # Make it the first element
        return ack