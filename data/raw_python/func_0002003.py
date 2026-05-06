def remove_link(self, rel, value=None, href=None):
        """
        Removes link nodes based on the function arguments.

        This can remove link nodes based on the following combinations of arguments:
            link/@rel
            link/@rel & link/text()
            link/@rel & link/@href
            link/@rel & link/text() & link/@href

        :param rel: link/@rel value to remove.  Required.
        :param value: link/text() value to remove. This is used in conjunction with link/@rel.
        :param href: link/@href value to remove. This is used in conjunction with link/@rel.
        :return: Return the number of link nodes removed, or False if no nodes are removed.
        """
        links_node = self.metadata.find('links')
        if links_node is None:
            log.warning('No links node present')
            return False
        counter = 0
        links = links_node.xpath('.//link[@rel="{}"]'.format(rel))
        for link in links:
            if value and href:
                if link.text == value and link.attrib['href'] == href:
                    links_node.remove(link)
                    counter += 1
            elif value and not href:
                if link.text == value:
                    links_node.remove(link)
                    counter += 1
            elif not value and href:
                if link.attrib['href'] == href:
                    links_node.remove(link)
                    counter += 1
            else:
                links_node.remove(link)
                counter += 1
        return counter