def _check_inclusions(self, f, domains=None):
        ''' Check file or directory against regexes in config to determine if
            it should be included in the index '''

        filename = f if isinstance(f, six.string_types) else f.path

        if domains is None:
            domains = list(self.domains.values())

        # Inject the Layout at the first position for global include/exclude
        domains = list(domains)
        domains.insert(0, self)

        for dom in domains:
            # If file matches any include regex, then True
            if dom.include:
                for regex in dom.include:
                    if re.search(regex, filename):
                        return True
                return False
            else:
                # If file matches any exclude regex, then False
                for regex in dom.exclude:
                    if re.search(regex, filename, flags=re.UNICODE):
                        return False
        return True