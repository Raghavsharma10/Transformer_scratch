def link_text(self):
        """
        Get a text represention of the links node.

        :return:
        """
        s = ''
        links_node = self.metadata.find('links')
        if links_node is None:
            return s
        links = links_node.getchildren()
        if links is None:
            return s
        s += 'IOC Links\n'
        for link in links:
            rel = link.attrib.get('rel', 'No Rel')
            href = link.attrib.get('href')
            text = link.text
            lt = '{rel}{href}: {text}\n'.format(rel=rel,
                                                href=' @ {}'.format(href) if href else '',
                                                text=text)
            s += lt
        return s