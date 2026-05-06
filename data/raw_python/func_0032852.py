def subdomain(self, hostname):
        """
        Determine of which known domain the given hostname is a subdomain.

        @return: A two-tuple giving the subdomain part and the domain part or
            C{None} if the domain is not a subdomain of any known domain.
        """
        hostname = hostname.split(":")[0]
        for domain in getDomainNames(self.siteStore):
            if hostname.endswith("." + domain):
                username = hostname[:-len(domain) - 1]
                if username != "www":
                    return username, domain
        return None