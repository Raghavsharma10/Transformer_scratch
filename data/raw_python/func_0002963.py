def make_article_info_footnotes_other(self, article_info_div):
        """
        This will catch all of the footnotes of type 'other' in the <fn-group>
        of the <back> element.
        """
        other_fn_expr = "./back/fn-group/fn[@fn-type='other']"
        other_fns = self.article.root.xpath(other_fn_expr)
        if other_fns:
            other_fn_div = etree.SubElement(article_info_div,
                                            'div',
                                            {'class': 'back-fn-other'})
        for other_fn in other_fns:
            append_all_below(other_fn_div, other_fn)