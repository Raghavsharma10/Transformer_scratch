def diff(self, report_a, report_b):
        """
        Generate a diff for two data reports.
        """
        arguments = GLOBAL_ARGUMENTS + ['run_date']

        output = OrderedDict([
            ('a', OrderedDict([(arg, report_a[arg]) for arg in arguments])),
            ('b', OrderedDict([(arg, report_b[arg]) for arg in arguments])),
            ('queries', [])
        ])

        output['a']

        for query_a in report_a['queries']:
            for query_b in report_b['queries']:
                if query_a['config'] == query_b['config']:
                    diff = OrderedDict()

                    diff['config'] = query_a['config']
                    diff['data_types'] = query_a['data_types']
                    diff['data'] = OrderedDict()

                    for metric, values in query_a['data'].items():
                        data_type = diff['data_types'][metric]
                        diff['data'][metric] = OrderedDict()

                        total_a = values['total']
                        total_b = query_b['data'][metric]['total']

                        for label, value in values.items():
                            a = value
                            
                            try:
                                b = query_b['data'][metric][label]
                            # TODO: hack for when labels are different...
                            except KeyError:
                                continue

                            change = b - a
                            percent_change = float(change) / a if a > 0 else None
                            
                            percent_a = float(a) / total_a if total_a > 0 else None
                            percent_b = float(b) / total_b if total_b > 0 else None

                            if label == 'total' or data_type == 'TIME' or percent_a is None or percent_b is None:
                                point_change = None
                            else:
                                point_change = percent_b - percent_a

                            diff['data'][metric][label] = OrderedDict([
                                ('change', change),
                                ('percent_change', percent_change),
                                ('point_change', point_change),
                            ])

                    output['queries'].append(diff)

            query_b = report_b['queries']

        return output