def get_simple_info_for_index(self, index=None, params={}, **kwargs):
        """
        Return a list of simple info by specified index (default all), each elements is a dictionary
        such as
        {
            'health' : 'green', 'status' : 'open',
            'index' : 'xxxx', 'uuid' : 'xxxx',
            'pri' : 1, 'rep' : 1,
            `docs_count` : 4, 'docs_deleted' : 0,
            'store_size' : 10kb, 'pri_store_size' : 10kb
        }
        """
        raw = self.client.cat.indices(index, params=params, **kwargs).split('\n')
        list = []
        for r in raw:
            alter = r.split(' ')
            if len(alter) < 10: continue
            dict = {
                'health': alter[0],
                'status': alter[1],
                'index': alter[2],
            }
            if len(alter) == 11:
                # May appear split fail (alter[3] is a empty string)
                dict['uuid'] = alter[4]
                i = 5
            else:
                dict['uuid'] = alter[3]
                i = 4
            dict['pri'] = alter[i]
            i += 1
            dict['rep'] = alter[i]
            i += 1
            dict['docs_count'] = alter[i]
            i += 1
            dict['docs_deleted'] = alter[i]
            i += 1
            dict['store_size'] = alter[i]
            i += 1
            dict['pri_store_size'] = alter[i]
            list.append(dict)
        logger.info('Acquire simple information of the index is done succeeded: %s' % len(list))
        return list