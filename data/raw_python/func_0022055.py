def query_domain(self, domain, typenames, domainquerytype='list', count=False):
        """
        Query by property domain values
        """

        objects = self._get_repo_filter(Layer.objects)

        if domainquerytype == 'range':
            return [tuple(objects.aggregate(Min(domain), Max(domain)).values())]
        else:
            if count:
                return [(d[domain], d['%s__count' % domain])
                        for d in objects.values(domain).annotate(Count(domain))]
            else:
                return objects.values_list(domain).distinct()