def nav_title(self, article):
        """
        Frontiers method.

        Given an Article class instance, this will return a string representing
        the title of the article. This is done for PloS by serializing the text
        in the Article's
        """
        article_title = article.metadata.front.article_meta.title_group.article_title.node
        return str(etree.tostring(article_title, method='text', encoding='utf-8'), encoding='utf-8')