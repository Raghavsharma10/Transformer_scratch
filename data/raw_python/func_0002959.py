def make_article_info_copyright(self, article_info_div):
        """
        Makes the copyright section for the ArticleInfo. For PLoS, this means
        handling the information contained in the metadata <permissions>
        element.
        """
        perm = self.article.root.xpath('./front/article-meta/permissions')
        if not perm:
            return
        copyright_div = etree.SubElement(article_info_div, 'div', {'id': 'copyright'})
        cp_bold = etree.SubElement(copyright_div, 'b')
        cp_bold.text = 'Copyright: '
        copyright_string = '\u00A9 '
        copyright_holder = perm[0].find('copyright-holder')
        if copyright_holder is not None:
            copyright_string += all_text(copyright_holder) + '. '
        lic = perm[0].find('license')
        if lic is not None:
            copyright_string += all_text(lic.find('license-p'))
        append_new_text(copyright_div, copyright_string)