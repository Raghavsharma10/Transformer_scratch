def make_back_glossary(self, body):
        """
        Glossaries are a fairly common item in papers for PLoS, but it also
        seems that they are rarely incorporated into the PLoS web-site or PDF
        formats. They are included in the ePub output however because they are
        helpful and because we can.
        """
        for glossary in self.article.root.xpath('./back/glossary'):
            gloss_copy = deepcopy(glossary)
            gloss_copy.tag = 'div'
            gloss_copy.attrib['class'] = 'back-glossary'
            body.append(gloss_copy)