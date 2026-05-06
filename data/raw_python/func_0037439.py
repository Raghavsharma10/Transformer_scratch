def _nodup_filter(self, min_hash, all_urls, max_sample=200):
        """ This filters results that are considered not duplicates.
        But we really need to check that, because lsh.query does not always
        return ALL duplicates, esp. when there are a lot of them, so
        here we double-check and return only urls that are NOT duplicates.
        Return estimated number of not duplicates.
        """
        if not all_urls:
            return 0
        urls = random.sample(all_urls, max_sample) \
               if len(all_urls) > max_sample else all_urls
        filtered = [
            url for url in urls
            if min_hash.jaccard(self.seen_urls[url].min_hash) <
            self.jaccard_threshold]
        return int(len(filtered) / len(urls) * len(all_urls))