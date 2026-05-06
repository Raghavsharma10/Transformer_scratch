def make_article_info_citation(self):
        """
        Creates a self citation node for the ArticleInfo of the article.

        This method uses code from this page as a reference implementation:
        https://github.com/PLOS/ambra/blob/master/base/src/main/resources/articleTransform-v3.xsl
        """
        citation_div = etree.Element('div', {'id': 'article-citation'})
        b = etree.SubElement(citation_div, 'b')
        b.text = 'Citation: '

        #Add author stuff to the citation
        authors = self.article.root.xpath("./front/article-meta/contrib-group/contrib[@contrib-type='author']")
        for author in authors:
            author_index = authors.index(author)
            #At the 6th author, simply append an et al., then stop iterating
            if author_index == 5:
                append_new_text(citation_div, 'et al.', join_str='')
                break
            else:
                #Check if the author contrib has a collab
                collab = author.find('collab')
                if collab is not None:
                    collab_copy = deepcopy(collab)
                    for contrib_group in collab_copy.findall('contrib_group'):
                        remove(contrib_group)
                    append_all_below(citation_div, collab, join_str='')
                else:  # Author element is not a collab
                    name = author.find('name')
                    #Note that this does not support eastern names
                    #Grab the surname information
                    surname = name.find('surname')
                    given_names = name.find('given-names')
                    suffix = name.find('suffix')
                    append_new_text(citation_div, surname.text, join_str='')
                    #Make initials from the given-name information
                    if given_names is not None:
                        #Add a space
                        append_new_text(citation_div, ' ', join_str='')
                        #Split by whitespace and take first character
                        given_initials = [i[0] for i in given_names.text.split() if i]
                        for initial in given_initials:
                            append_new_text(citation_div, initial, join_str='')
                    #If there is a suffix, add its text, but don't include the
                    #trailing period if there is one
                    if suffix is not None:
                        #Add a space
                        append_new_text(citation_div, ' ', join_str='')
                        suffix_text = suffix.text
                        #Check for the trailing period
                        if suffix_text[-1] == '.':
                            suffix_text = suffix_text[:-1]
                        append_new_text(citation_div, suffix_text, join_str='')
                #If this is not the last author to be added, add a ", "
                #This is satisfied by being less than the 6th author, or less
                #than the length of the author_list - 1
                if author_index < 5 or author_index < len(author_list) -1:
                    append_new_text(citation_div, ', ', join_str='')
        #Add Publication Year to the citation
        #Find pub-date elements, use pub-type=collection, or else pub-type=ppub
        d = './front/article-meta/pub-date'
        coll = self.article.root.xpath(d + "[@pub-type='collection']")
        ppub = self.article.root.xpath(d + "[@pub-type='ppub']")
        if coll:
            pub_year = coll[0].find('year').text
        elif ppub:
            pub_year = ppub[0].find('year').text
        append_new_text(citation_div, ' ({0}) '.format(pub_year), join_str='')
        #Add the Article Title to the Citation
        #As best as I can tell from the reference implementation, they
        #serialize the article title to text-only, and expunge redundant spaces
        #This might need later review
        article_title = self.article.root.xpath('./front/article-meta/title-group/article-title')[0]
        article_title_text = serialize(article_title)
        normalized = ' '.join(article_title_text.split())  # Remove redundant whitespace
        #Add a period unless there is some other valid punctuation
        if normalized[-1] not in '.?!':
            normalized += '.'
        append_new_text(citation_div, normalized + ' ', join_str='')
        #Add the article's journal name using the journal-id of type "nlm-ta"
        journal = self.article.root.xpath("./front/journal-meta/journal-id[@journal-id-type='nlm-ta']")
        append_new_text(citation_div, journal[0].text + ' ', join_str='')
        #Add the article's volume, issue, and elocation_id  values
        volume = self.article.root.xpath('./front/article-meta/volume')[0].text
        issue = self.article.root.xpath('./front/article-meta/issue')[0].text
        elocation_id = self.article.root.xpath('./front/article-meta/elocation-id')[0].text
        form = '{0}({1}): {2}. '.format(volume, issue, elocation_id)
        append_new_text(citation_div, form, join_str='')
        append_new_text(citation_div, 'doi:{0}'.format(self.article.doi), join_str='')

        return citation_div