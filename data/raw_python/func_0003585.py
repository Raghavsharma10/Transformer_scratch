def add_chapter(self, c):
        """
        Add a Chapter to your epub.

        Args:
            c (Chapter): A Chapter object representing your chapter.

        Raises:
            TypeError: Raised if a Chapter object isn't supplied to this
                method.
        """
        try:
            assert type(c) == chapter.Chapter
        except AssertionError:
            raise TypeError('chapter must be of type Chapter')
        chapter_file_output = os.path.join(self.OEBPS_DIR, self.current_chapter_path)
        c._replace_images_in_chapter(self.OEBPS_DIR)
        c.write(chapter_file_output)
        self._increase_current_chapter_number()
        self.chapters.append(c)