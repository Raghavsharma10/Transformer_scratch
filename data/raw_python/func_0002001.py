def update_link_rewrite(self, old_rel, old_text, new_text, single_link=False):
        """
        Rewrite the text() value of a link based on the link/@rel and link/text() value.

        This is similar to update_link_rel_based but users link/@rel AND link/text() values
        to determine which links have their link/@text() values updated.

        :param old_rel: The link/@rel value used to select link nodes to update.
        :param old_text: The link/text() value used to select link nodes to update.
        :param new_text: The new link/text() value to set on link nodes.
        :param single_link: Determine if only the first, or multiple, linkes are modified.
        :return: True, unless there are no links with link/[@rel='old_rel' and text()='old_text']
        """
        links = self.metadata.xpath('./links/link[@rel="{}" and text()="{}"]'.format(old_rel, old_text))
        if len(links) < 1:
            log.warning('No links with link/[@rel="{}"and text()="{}"]'.format(str(old_rel), str(old_text)))
            return False
        for link in links:
            link.text = new_text
            if single_link:
                break
        return True