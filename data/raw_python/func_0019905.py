def categorization(self, domains, labels=False):
        '''Get the domain status and categorization of a domain or list of domains.
        'domains' can be either a single domain, or a list of domains.
        Setting 'labels' to True will give back categorizations in human-readable
        form.

        For more detail, see https://investigate.umbrella.com/docs/api#categorization
        '''
        if type(domains) is str:
            return self._get_categorization(domains, labels)
        elif type(domains) is list:
            return self._post_categorization(domains, labels)
        else:
            raise Investigate.DOMAIN_ERR