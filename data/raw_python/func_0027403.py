def _compile_references(self, url, tree):
        '''
        Returns a list of catalog reference URLs for the current catalog
        :param str url: URL for the current catalog
        :param lxml.etree.Eleemnt tree: Current XML Tree
        '''
        references = []
        for ref in tree.findall('.//{%s}catalogRef' % INV_NS):
            # Check skips
            title = ref.get("{%s}title" % XLINK_NS)
            if any([x.match(title) for x in self.skip]):
                logger.info("Skipping catalogRef based on 'skips'.  Title: %s" % title)
                continue
            references.append(construct_url(url, ref.get("{%s}href" % XLINK_NS)))
        return references