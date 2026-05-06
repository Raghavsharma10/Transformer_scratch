def items(self):
        ''' Lista dos items do pagamento
        '''
        if type(self.transaction['items']['item']) == list:
            return self.transaction['items']['item']
        else:
            return [self.transaction['items']['item'],]