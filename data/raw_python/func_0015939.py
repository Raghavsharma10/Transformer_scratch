def column_search(self, queryset):
        '''Filter a queryset with column search'''
        for idx in xrange(self.dt_data['iColumns']):
            search = self.dt_data['sSearch_%s' % idx]
            if search:
                if hasattr(self, 'search_col_%s' % idx):
                    custom_search = getattr(self, 'search_col_%s' % idx)
                    queryset = custom_search(search, queryset)
                else:
                    field = self.get_field(idx)
                    fields = RE_FORMATTED.findall(field) if RE_FORMATTED.match(field) else [field]
                    if self.dt_data['bRegex_%s' % idx]:
                        criterions = [Q(**{'%s__iregex' % field: search}) for field in fields if self.can_regex(field)]
                        if len(criterions) > 0:
                            search = reduce(or_, criterions)
                            queryset = queryset.filter(search)
                    else:
                        for term in search.split():
                            criterions = (Q(**{'%s__icontains' % field: term}) for field in fields)
                            search = reduce(or_, criterions)
                            queryset = queryset.filter(search)
        return queryset