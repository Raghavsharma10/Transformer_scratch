def create_chapter_from_string(self, html_string, url=None, title=None):
        """
        Creates a Chapter object from a string. Sanitizes the
        string using the clean_function method, and saves
        it as the content of the created chapter.

        Args:
            html_string (string): The html or xhtml content of the created
                Chapter
            url (Option[string]): A url to infer the title of the chapter from
            title (Option[string]): The title of the created Chapter. By
                default, this is None, in which case the title will try to be
                inferred from the webpage at the url.

        Returns:
            Chapter: A chapter object whose content is the given string
                and whose title is that provided or inferred from the url
        """
        clean_html_string = self.clean_function(html_string)
        clean_xhtml_string = clean.html_to_xhtml(clean_html_string)
        if title:
            pass
        else:
            try:
                root = BeautifulSoup(html_string, 'html.parser')
                title_node = root.title
                if title_node is not None:
                    title = unicode(title_node.string)
                else:
                    raise ValueError
            except (IndexError, ValueError):
                title = 'Ebook Chapter'
        return Chapter(clean_xhtml_string, title, url)