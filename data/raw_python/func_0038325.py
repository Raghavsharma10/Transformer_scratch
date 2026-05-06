def extract_pagination_links(self, response):
        '''Given a wrapped_response from a Canvas API endpoint,
        extract the pagination links from the response headers'''
        try:
            link_header = response.headers['Link']
        except KeyError:
            logger.warn('Unable to find the Link header. Unable to continue with pagination.')
            return None

        split_header = link_header.split(',')
        exploded_split_header = [i.split(';') for i in split_header]

        pagination_links = {}
        for h in exploded_split_header:
            link = h[0]
            rel = h[1]
            # Check that the link format is what we expect
            if link.startswith('<') and link.endswith('>'):
                link = link[1:-1]
            else:
                continue
            # Extract the rel argument
            m = self.rel_matcher.match(rel)
            try:
                rel = m.groups()[0]
            except AttributeError:
                # Match returned None, just skip.
                continue
            except IndexError:
                # Matched but no groups returned
                continue

            pagination_links[rel] = link
        return pagination_links