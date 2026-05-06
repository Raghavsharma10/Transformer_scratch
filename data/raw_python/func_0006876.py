def invert(self):
      ''' Return inverse mapping of dictionary with sorted values.
      USAGE
         >>> # Switch the keys and values
         >>> adv_dict({
         ...     'A': [1, 2, 3],
         ...     'B': [4, 2],
         ...     'C': [1, 4],
         ... }).invert()
         {1: ['A', 'C'], 2: ['A', 'B'], 3: ['A'], 4: ['B', 'C']}
      '''
      inv_map = {}
      for k, v in self.items():
         if sys.version_info < (3, 0):
            acceptable_v_instance = isinstance(v, (str, int, float, long))
         else:
            acceptable_v_instance = isinstance(v, (str, int, float))
         if acceptable_v_instance: v = [v]
         elif not isinstance(v, list):
            raise Exception('Error: Non supported value format! Values may only'
                            ' be numerical, strings, or lists of numbers and '
                            'strings.')
         for val in v:
            inv_map[val] = inv_map.get(val, [])
            inv_map[val].append(k)
            inv_map[val].sort()
      return inv_map