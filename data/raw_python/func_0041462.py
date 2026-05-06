def _queue(self, kwargs):
    """The hard resource_list comes like this: '<qname>=TRUE,mem=128M'. To
    process it we have to split it twice (',' and then on '='), create a
    dictionary and extract just the qname"""
    if not 'hard resource_list' in kwargs: return 'all.q'
    d = dict([k.split('=') for k in kwargs['hard resource_list'].split(',')])
    for k in d:
      if k[0] == 'q' and d[k] == 'TRUE': return k
    return 'all.q'