def get_orders(self):
        '''Get ordering fields for ``QuerySet.order_by``'''
        orders = []
        iSortingCols = self.dt_data['iSortingCols']
        dt_orders = [(self.dt_data['iSortCol_%s' % i], self.dt_data['sSortDir_%s' % i]) for i in xrange(iSortingCols)]
        for field_idx, field_dir in dt_orders:
            direction = '-' if field_dir == DESC else ''
            if hasattr(self, 'sort_col_%s' % field_idx):
                method = getattr(self, 'sort_col_%s' % field_idx)
                result = method(direction)
                if isinstance(result, (bytes, text_type)):
                    orders.append(result)
                else:
                    orders.extend(result)
            else:
                field = self.get_field(field_idx)
                if RE_FORMATTED.match(field):
                    tokens = RE_FORMATTED.findall(field)
                    orders.extend(['%s%s' % (direction, token) for token in tokens])
                else:
                    orders.append('%s%s' % (direction, field))
        return orders