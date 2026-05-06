def _yield_leaves(self, url, tree):
        '''
        Yields a URL corresponding to a leaf dataset for each dataset described by the catalog
        :param str url: URL for the current catalog
        :param lxml.etree.Eleemnt tree: Current XML Tree
        '''
        for leaf in tree.findall('.//{%s}dataset[@urlPath]' % INV_NS):
            # Subset by the skips
            name = leaf.get("name")
            if any([x.match(name) for x in self.skip]):
                logger.info("Skipping dataset based on 'skips'.  Name: %s" % name)
                continue

            # Subset by before and after
            date_tag = leaf.find('.//{%s}date[@type="modified"]' % INV_NS)
            if date_tag is not None:
                try:
                    dt = parse(date_tag.text)
                except ValueError:
                    logger.error("Skipping dataset.Wrong date string %s " % date_tag.text)
                    continue
                else:
                    dt = dt.replace(tzinfo=pytz.utc)
                if self.after and dt < self.after:
                    continue
                if self.before and dt > self.before:
                    continue

            # Subset by the Selects defined
            gid = leaf.get('ID')
            if self.select is not None:
                if gid is not None and any([x.match(gid) for x in self.select]):
                    logger.debug("Processing %s" % gid)
                    yield "%s?dataset=%s" % (url, gid)
                else:
                    logger.info("Ignoring dataset based on 'selects'.  ID: %s" % gid)
                    continue
            else:
                logger.debug("Processing %s" % gid)
                yield "%s?dataset=%s" % (url, gid)