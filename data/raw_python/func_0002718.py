def has_out_of_flow_tables(self):
        """
        Returns True if the article has out-of-flow tables, indicates separate
        tables document.

        This method is used to indicate whether rendering this article's content
        will result in the creation of out-of-flow HTML tables. This method has
        a base class implementation representing a common logic; if an article
        has a graphic(image) representation of a table then the HTML
        representation will be placed out-of-flow if it exists, if there is no
        graphic(image) represenation then the HTML representation will be placed
        in-flow.

        Returns
        -------
        bool
            True if there are out-of-flow HTML tables, False otherwise
        """
        if self.article.body is None:
            return False
        for table_wrap in self.article.body.findall('.//table-wrap'):
            graphic = table_wrap.xpath('./graphic | ./alternatives/graphic')
            table = table_wrap.xpath('./table | ./alternatives/table')
            if graphic and table:
                return True
        return False