def make_article_info(self):
        """
        The Article Info contains the (self) Citation, Editors, Dates,
        Copyright, Funding Statement, Competing Interests Statement,
        Correspondence, and Footnotes. Maybe more...

        This content follows the Heading and precedes the Main segment in the
        output.

        This function accepts the receiving_node argument, which will receive
        all generated output as new childNodes.
        """
        body = self.main.getroot().find('body')
        #Create a div for ArticleInfo, exposing it to linking and formatting
        article_info_div = etree.Element('div', {'id': 'ArticleInfo'})
        body.insert(1, article_info_div)
        #Creation of the self Citation
        article_info_div.append(self.make_article_info_citation())
        #Creation of the Editors
        editors = self.article.root.xpath("./front/article-meta/contrib-group/contrib[@contrib-type='editor']")
        self.make_article_info_editors(editors, article_info_div)
        #Creation of the important Dates segment
        article_info_div.append(self.make_article_info_dates())
        #Creation of the Copyright statement
        self.make_article_info_copyright(article_info_div)
        #Creation of the Funding statement
        self.make_article_info_funding(article_info_div)
        #Creation of the Competing Interests statement
        self.make_article_info_competing_interests(article_info_div)
        #Creation of the Correspondences (contact information) for the article
        self.make_article_info_correspondences(article_info_div)
        #Creation of the Footnotes (other) for the ArticleInfo
        self.make_article_info_footnotes_other(article_info_div)