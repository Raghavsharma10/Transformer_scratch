def _get_or_load_domain(self, domain):
        ''' Return a domain if one already exists, or create a new one if not.

        Args:
            domain (str, dict): Can be one of:
                - The name of the Domain to return (fails if none exists)
                - A path to the Domain configuration file
                - A dictionary containing configuration information
        '''
        if isinstance(domain, six.string_types):
            if domain in self.domains:
                return self.domains[domain]
            elif exists(domain):
                with open(domain, 'r') as fobj:
                    domain = json.load(fobj)
            else:
                raise ValueError("No domain could be found/loaded from input "
                                 "'{}'; value must be either the name of an "
                                 "existing Domain, or a valid path to a "
                                 "configuration file.".format(domain))

        # At this point, domain is a dict
        name = domain['name']
        if name in self.domains:
            msg = ("Domain with name '{}' already exists; returning existing "
                   "Domain configuration.".format(name))
            warnings.warn(msg)
            return self.domains[name]

        entities = domain.get('entities', [])
        domain = Domain(domain)
        for e in entities:
            self.add_entity(domain=domain, **e)
        self.domains[name] = domain
        return self.domains[name]