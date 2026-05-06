def removeInput(self, key_id, input_type='key'):
        """Remove key (key, value, map) from Input
        key_id : id of the input element i.e <key id='artist' />
        input_type : type of the input ; key, value or map
        """
        root = self.etree
        t_inputs = root.find('inputs')

        if not t_inputs:
            return False

        keys = t_inputs.findall(input_type)

        key = [ key for key in keys if key.get('id') == key_id ]

        try:
            t_inputs.remove(key[0])
            return True
        except (Exception,) as e:
            print(e)

        return False