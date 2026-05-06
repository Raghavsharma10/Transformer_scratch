def resolve_url(self, catalog_url):
        """Resolve the url of the dataset when reading latest.xml.

        Parameters
        ----------
        catalog_url : str
            The catalog url to be resolved

        """
        if catalog_url != '':
            resolver_base = catalog_url.split('catalog.xml')[0]
            resolver_url = resolver_base + self.url_path
            resolver_xml = session_manager.urlopen(resolver_url)
            tree = ET.parse(resolver_xml)
            root = tree.getroot()
            if 'name' in root.attrib:
                self.catalog_name = root.attrib['name']
            else:
                self.catalog_name = 'No name found'
            resolved_url = ''
            found = False
            for child in root.iter():
                if not found:
                    tag_type = child.tag.split('}')[-1]
                    if tag_type == 'dataset':
                        if 'urlPath' in child.attrib:
                            ds = Dataset(child)
                            resolved_url = ds.url_path
                            found = True
            if found:
                return resolved_url
            else:
                log.warning('no dataset url path found in latest.xml!')