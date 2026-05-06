def diff_column(self, column1, column2):
        """
        Returns the difference between column1 and column2

        :type column1: eloquent.dbal.column.Column
        :type column2: eloquent.dbal.column.Column

        :rtype: list
        """
        properties1 = column1.to_dict()
        properties2 = column2.to_dict()

        changed_properties = []

        for prop in ['type', 'notnull', 'unsigned', 'autoincrement']:
            if properties1[prop] != properties2[prop]:
                changed_properties.append(prop)

        if properties1['default'] != properties2['default']\
                or (properties1['default'] is None and properties2['default'] is not None)\
                or (properties2['default'] is None and properties1['default'] is not None):
            changed_properties.append('default')

        if properties1['type'] == 'string' and properties1['type'] != 'guid'\
                or properties1['type'] in ['binary', 'blob']:
            length1 = properties1['length'] or 255
            length2 = properties2['length'] or 255

            if length1 != length2:
                changed_properties.append('length')

            if properties1['fixed'] != properties2['fixed']:
                changed_properties.append('fixed')
        elif properties1['type'] in ['decimal', 'float', 'double precision']:
            precision1 = properties1['precision'] or 10
            precision2 = properties2['precision'] or 10

            if precision1 != precision2:
                changed_properties.append('precision')

            if properties1['scale'] != properties2['scale']:
                changed_properties.append('scale')

        return list(set(changed_properties))