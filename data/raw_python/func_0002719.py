def process(self, article):
        """
        Ingests an Article to create navigation structures and parse global
        metadata.
        """
        if self.article is not None and not self.collection:
            log.warning('Could not process additional article. Navigation only \
handles one article unless collection mode is set.')
            return False

        if article.publisher is None:
            log.error('''Navigation cannot be generated for an Article \
without a publisher!''')
            return
        self.article = article
        self.article_doi = self.article.doi.split('/')[1]
        self.all_dois.append(self.article.doi)
        if self.collection:
            pass
        else:
            self.title = self.article.publisher.nav_title()
        for author in self.article.publisher.nav_contributors():
            self.contributors.add(author)

        #Analyze the structure of the article to create internal mapping
        self.map_navigation()