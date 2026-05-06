def query(self, constraint, sortby=None, typenames=None, maxrecords=10, startposition=0):
        """
        Query records from underlying repository
        """

        # run the raw query and get total
        # we want to exclude layers which are not valid, as it is done in the search engine
        if 'where' in constraint:  # GetRecords with constraint
            query = self._get_repo_filter(Layer.objects).filter(
                is_valid=True).extra(where=[constraint['where']], params=constraint['values'])
        else:  # GetRecords sans constraint
            query = self._get_repo_filter(Layer.objects).filter(is_valid=True)

        total = query.count()

        # apply sorting, limit and offset
        if sortby is not None:
            if 'spatial' in sortby and sortby['spatial']:  # spatial sort
                desc = False
                if sortby['order'] == 'DESC':
                    desc = True
                query = query.all()
                return [str(total),
                        sorted(query,
                               key=lambda x: float(util.get_geometry_area(getattr(x, sortby['propertyname']))),
                               reverse=desc,
                               )[startposition:startposition+int(maxrecords)]]
            else:
                if sortby['order'] == 'DESC':
                    pname = '-%s' % sortby['propertyname']
                else:
                    pname = sortby['propertyname']
                return [str(total),
                        query.order_by(pname)[startposition:startposition+int(maxrecords)]]
        else:  # no sort
            return [str(total), query.all()[startposition:startposition+int(maxrecords)]]