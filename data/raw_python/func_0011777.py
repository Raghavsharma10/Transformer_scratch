def process_results(self):
        """ Process results by providers """
        for result in self._results:
            provider = result.provider
            self.providers.append(provider)
            if result.error:
                self.failed_providers.append(provider)
                continue
            if not result.response:
                continue
            # set blacklisted to True if ip is detected with at least one dnsbl
            self.blacklisted = True
            provider_categories = provider.process_response(result.response)
            assert provider_categories.issubset(DNSBL_CATEGORIES)
            self.categories = self.categories.union(provider_categories)
            self.detected_by[provider.host] = list(provider_categories)