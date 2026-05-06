def package_description(self):
        """
        Given an Article class instance, this is responsible for returning an
        article description. For this method I have taken the approach of
        serializing the article's first abstract, if it has one. This results
        in 0 or 1 descriptions per article.
        """
        abstract = self.article.root.xpath('./front/article-meta/abstract')
        return serialize(abstract[0], strip=True) if abstract else None