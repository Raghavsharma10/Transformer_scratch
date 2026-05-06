def crawl_links(self, seed_url=None):
        """Find new links given a seed URL and follow them breadth-first.

        Save page responses as PART.html files.
        Return the PART.html filenames created during crawling.
        """
        if seed_url is not None:
            self.seed_url = seed_url

        if self.seed_url is None:
            sys.stderr.write('Crawling requires a seed URL.\n')
            return []

        prev_part_num = utils.get_num_part_files()
        crawled_links = set()
        uncrawled_links = OrderedSet()

        uncrawled_links.add(self.seed_url)
        try:
            while uncrawled_links:
                # Check limit on number of links and pages to crawl
                if self.limit_reached(len(crawled_links)):
                    break
                url = uncrawled_links.pop(last=False)

                # Remove protocol, fragments, etc. to get unique URLs
                unique_url = utils.remove_protocol(utils.clean_url(url))
                if unique_url not in crawled_links:
                    raw_resp = utils.get_raw_resp(url)
                    if raw_resp is None:
                        if not self.args['quiet']:
                            sys.stderr.write('Failed to parse {0}.\n'.format(url))
                        continue

                    resp = lh.fromstring(raw_resp)
                    if self.page_crawled(resp):
                        continue

                    crawled_links.add(unique_url)
                    new_links = self.get_new_links(url, resp)
                    uncrawled_links.update(new_links)
                    if not self.args['quiet']:
                        print('Crawled {0} (#{1}).'.format(url, len(crawled_links)))

                    # Write page response to PART.html file
                    utils.write_part_file(self.args, url, raw_resp, resp, len(crawled_links))
        except (KeyboardInterrupt, EOFError):
            pass

        curr_part_num = utils.get_num_part_files()
        return utils.get_part_filenames(curr_part_num, prev_part_num)