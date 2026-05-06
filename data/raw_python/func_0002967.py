def make_back_author_contributions(self, body):
        """
        Though this goes in the back of the document with the rest of the back
        matter, it is not an element found under <back>.

        I don't expect to see more than one of these. Compare this method to
        make_article_info_competing_interests()
        """
        cont_expr = "./front/article-meta/author-notes/fn[@fn-type='con']"
        contribution = self.article.root.xpath(cont_expr)
        if contribution:
            author_contrib = deepcopy(contribution[0])
            remove_all_attributes(author_contrib)
            author_contrib.tag = 'div'
            author_contrib.attrib['id'] = 'author-contributions'
            #This title element will be parsed later
            title = etree.Element('title')
            title.text = 'Author Contributions'
            author_contrib.insert(0, title)
            body.append(author_contrib)