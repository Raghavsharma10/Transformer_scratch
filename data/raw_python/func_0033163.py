def _extract(self):
        """Extract email addresses from results.

        Text content from all crawled pages are ran through a simple email
        extractor. Data is cleaned prior to running pattern expressions.
        """
        self.log.debug("Extracting emails from text content")
        for item in self.data:
            emails = extract_emails(item, self.domain, self.fuzzy)
            self.results.extend(emails)
        self.log.debug("Email extraction completed")
        return list(set(self.results))