def instant_articles(self, **kwargs):
        """
        QuerySet including all published content approved for instant articles.

        Instant articles are configured via FeatureType. FeatureType.instant_article = True.
        """
        eqs = self.search(**kwargs).sort('-last_modified', '-published')
        return eqs.filter(InstantArticle())