def update_link_rel_based(self, old_rel, new_rel=None, new_text=None, single_link=False):
        """
        Update link nodes, based on the existing link/@rel values.

        This requires specifying a link/@rel value to update, and either a new
        link/@rel value, or a new link/text() value for all links which match
        the link/@rel value.  Optionally, only the first link which matches the
        link/@rel value will be modified.

        :param old_rel: The link/@rel value used to select link nodes to update
        :param new_rel: The new link/@rel value
        :param new_text: The new link/text() value
        :param single_link: Determine if only the first, or multiple, linkes are modified.
        :return: True, unless there are no links with link[@rel='old_rel']
        """
        links = self.metadata.xpath('./links/link[@rel="{}"]'.format(old_rel))
        if len(links) < 1:
            log.warning('No links with link/[@rel="{}"]'.format(str(old_rel)))
            return False
        if new_rel and not new_text:
            # update link/@rel value
            for link in links:
                link.attrib['rel'] = new_rel
                if single_link:
                    break
        elif not new_rel and new_text:
            # update link/@text() value
            for link in links:
                link.text = new_text
                if single_link:
                    break
        elif new_rel and new_text:
            log.warning('Cannot update rel and text at the same time')
            return False
        else:
            log.warning('Must specify either new_rel or new_text arguments')
            return False
        return True