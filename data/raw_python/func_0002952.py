def heading_title(self):
        """
        Makes the Article Title for the Heading.

        Metadata element, content derived from FrontMatter
        """
        art_title = self.article.root.xpath('./front/article-meta/title-group/article-title')[0]
        article_title = deepcopy(art_title)
        article_title.tag = 'h1'
        article_title.attrib['id'] = 'title'
        article_title.attrib['class'] = 'article-title'
        return article_title