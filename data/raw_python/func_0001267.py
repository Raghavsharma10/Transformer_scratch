def get_report_order(self):
        ''' Keys are sorted based on report order (i.e. some keys to be shown first)
            Related: see sorted_by_count
        '''
        order_list = []
        for x in self.__priority:
            order_list.append([x, self[x]])
        for x in sorted(list(self.keys())):
            if x not in self.__priority:
                order_list.append([x, self[x]])
        return order_list