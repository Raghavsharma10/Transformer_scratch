def make_article_info_correspondences(self, article_info_div):
        """
        Articles generally provide a first contact, typically an email address
        for one of the authors. This will supply that content.
        """
        corresps = self.article.root.xpath('./front/article-meta/author-notes/corresp')
        if corresps:
            corresp_div = etree.SubElement(article_info_div,
                                           'div',
                                           {'id': 'correspondence'})
        for corresp in corresps:
            sub_div = etree.SubElement(corresp_div,
                                       'div',
                                       {'id': corresp.attrib['id']})
            append_all_below(sub_div, corresp)