def get_from_solr(clone, number, row_size):
        """
        If not start parameter is given, with the given number(0,1,2..) multiplies default row size
        and determines start parameter. Takes results from solr according to this parameter. For
        example, if number is 2 and default row size is 1000, takes results from solr between 2000
        and 3000. But if start parameter is given, start value is found adding given start paramater.
        For example start paramater is given as 550, if number is 2 and default row size is 1000,
        takes results from solr between 2550 and 3550.

        Args:
            clone: Queryset adapter clone
            number(int): Uses for solr start parameter. Multiplies with default row size.
            row_size(int): Uses for solr rows parameter. Indicates how many record will be taken
                           from solr.

        Returns:
             tuple with given number and riak_multi_get method input list.
             Example return = (0, [('models','personel','McAPchPZzB6RVJ8QI2XSVQk4mUR'),
                                 ('models','personel','XyZZrsadVJ8QI2XSVQk4mUR'),
                                 ('models','personel','SkFl3RPZzB6RVJ8QI2XSVQk4mUR'),
                                 ('models','personel','PxCdytPZzB6RVJ8QI2XSVQk4mUR')])

        """
        start = number * clone._cfg['row_size'] + clone._cfg['start']
        clone._solr_params.update({'start': start})
        clone._solr_params.update({'rows': row_size})
        clone._solr_locked = False
        return number, [(clone._cfg['bucket_type'], clone._cfg['bucket_name'],
                         ub_to_str(doc.get('_yz_rk'))) for doc in clone._exec_query()]