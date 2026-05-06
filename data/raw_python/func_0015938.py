def global_search(self, queryset):
        '''Filter a queryset with global search'''
        search = self.dt_data['sSearch']
        if search:
            if self.dt_data['bRegex']:
                criterions = [
                    Q(**{'%s__iregex' % field: search})
                    for field in self.get_db_fields()
                    if self.can_regex(field)
                ]
                if len(criterions) > 0:
                    search = reduce(or_, criterions)
                    queryset = queryset.filter(search)
            else:
                for term in search.split():
                    criterions = (Q(**{'%s__icontains' % field: term}) for field in self.get_db_fields())
                    search = reduce(or_, criterions)
                    queryset = queryset.filter(search)
        return queryset