def dropna(self):
        """
        :return:

        """
        step = {
            'data-set': self.iid,
            'operation': 'drop-na',
            'expression': '{"axis": 0}'
        }

        self.attr_update(attr='steps', value=[step])