def get_tree(self, list_of_keys):
      """ gettree will extract the value from a nested tree
      
      INPUT
         list_of_keys: a list of keys ie. ['key1', 'key2']
      USAGE
      >>> # Access the value for key2 within the nested dictionary
      >>> adv_dict({'key1': {'key2': 'value'}}).gettree(['key1', 'key2'])
      'value'
      """
      cur_obj = self
      for key in list_of_keys:
         cur_obj = cur_obj.get(key)
         if not cur_obj: break
      return cur_obj