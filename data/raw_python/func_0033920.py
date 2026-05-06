def set_dn(self,dn):
    """
    Use the domain column in the process table to store the DN

    dn = dn to be added
    """
    try:
      domain_col = self.table['process']['orderedcol'].index('domain')
      for row_idx in range(len(self.table['process']['stream'])):
        row_list = list(self.table['process']['stream'][row_idx])
        row_list[domain_col] = dn
        self.table['process']['stream'][row_idx] = tuple(row_list)
    except ValueError:
      self.table['process']['column']['domain'] = 'lstring'
      self.table['process']['orderedcol'].append('domain')
      for row_idx in range(len(self.table['process']['stream'])):
        row_list = list(self.table['process']['stream'][row_idx])
        row_list.append(dn)
        self.table['process']['stream'][row_idx] = tuple(row_list)