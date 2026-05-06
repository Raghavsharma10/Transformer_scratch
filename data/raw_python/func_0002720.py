def map_navigation(self):
        """
        This is a wrapper for depth-first recursive analysis of the article
        """
        #All articles should have titles
        title_id = 'titlepage-{0}'.format(self.article_doi)
        title_label = self.article.publisher.nav_title()
        title_source = 'main.{0}.xhtml#title'.format(self.article_doi)
        title_navpoint = navpoint(title_id, title_label, self.play_order,
                                  title_source, [])
        self.nav.append(title_navpoint)
        #When processing a collection of articles, we will want all subsequent
        #navpoints for this article to be located under the title
        if self.collection:
            nav_insertion = title_navpoint.children
        else:
            nav_insertion = self.nav

        #If the article has a body, we'll need to parse it for navigation
        if self.article.body is not None:
            #Here is where we invoke the recursive parsing!
            for nav_pt in self.recursive_article_navmap(self.article.body):
                nav_insertion.append(nav_pt)

        #Add a navpoint to the references if appropriate
        if self.article.root.xpath('./back/ref'):
            ref_id = 'references-{0}'.format(self.article_doi)
            ref_label = 'References'
            ref_source = 'biblio.{0}.xhtml#references'.format(self.article_doi)
            ref_navpoint = navpoint(ref_id, ref_label, self.play_order,
                                    ref_source, [])
            nav_insertion.append(ref_navpoint)