def create_chapter_from_file(self, file_name, url=None, title=None):
        """
        Creates a Chapter object from an html or xhtml file. Sanitizes the
        file's content using the clean_function method, and saves
        it as the content of the created chapter.

        Args:
            file_name (string): The file_name containing the html or xhtml
                content of the created Chapter
            url (Option[string]): A url to infer the title of the chapter from
            title (Option[string]): The title of the created Chapter. By
                default, this is None, in which case the title will try to be
                inferred from the webpage at the url.

        Returns:
            Chapter: A chapter object whose content is the given file
                and whose title is that provided or inferred from the url
        """
        with codecs.open(file_name, 'r', encoding='utf-8') as f:
            content_string = f.read()
        return self.create_chapter_from_string(content_string, url, title)