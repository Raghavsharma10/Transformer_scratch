def scan(self, concurrency=1):
        """Iterates over the applications installed within the disk
        and queries the CVE DB to determine whether they are vulnerable.

        Concurrency controls the amount of concurrent queries
        against the CVE DB.

        For each vulnerable application the method yields a namedtuple:

        VulnApp(name             -> application name
                version          -> application version
                vulnerabilities) -> list of Vulnerabilities

        Vulnerability(id       -> CVE Id
                      summary) -> brief description of the vulnerability

        """
        self.logger.debug("Scanning FS content.")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = executor.map(self.query_vulnerabilities,
                                   self.applications())

        for report in results:
            application, vulnerabilities = report
            vulnerabilities = list(lookup_vulnerabilities(application.version,
                                                          vulnerabilities))

            if vulnerabilities:
                yield VulnApp(application.name,
                              application.version,
                              vulnerabilities)