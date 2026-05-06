def toc_directive(self, maxdepth=1):
        """
        Generate toctree directive text.

        :param table_of_content_header:
        :param header_bar_char:
        :param header_line_length:
        :param maxdepth:
        :return:
        """
        articles_directive_content = TC.toc.render(
            maxdepth=maxdepth,
            article_list=self.sub_article_folders,
        )
        return articles_directive_content