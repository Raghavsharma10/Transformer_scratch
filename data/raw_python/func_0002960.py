def make_article_info_funding(self, article_info_div):
        """
        Creates the element for declaring Funding in the article info.
        """
        funding_group = self.article.root.xpath('./front/article-meta/funding-group')
        if funding_group:
            funding_div = etree.SubElement(article_info_div,
                                           'div',
                                           {'id': 'funding'})
            funding_b = etree.SubElement(funding_div, 'b')
            funding_b.text = 'Funding: '
            #As far as I can tell, PLoS only uses one funding-statement
            funding_statement = funding_group[0].find('funding-statement')
            append_all_below(funding_div, funding_statement)