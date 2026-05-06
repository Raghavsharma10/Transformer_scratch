def csv_line_items(self):
        '''
        Invoices from lists omit csv-line-items

        '''
        if not hasattr(self, '_csv_line_items'):
            url = '{}/{}'.format(self.base_url, self.id)
            self._csv_line_items = self.harvest._get_element_values(url, self.element_name).next().get('csv-line-items', '')
        return self._csv_line_items