def make_article_info_competing_interests(self, article_info_div):
        """
        Creates the element for declaring competing interests in the article
        info.
        """
        #Check for author-notes
        con_expr = "./front/article-meta/author-notes/fn[@fn-type='conflict']"
        conflict = self.article.root.xpath(con_expr)
        if not conflict:
            return
        conflict_div = etree.SubElement(article_info_div,
                                        'div',
                                        {'id': 'conflict'})
        b = etree.SubElement(conflict_div, 'b')
        b.text = 'Competing Interests: '
        fn_p = conflict[0].find('p')
        if fn_p is not None:
            append_all_below(conflict_div, fn_p)