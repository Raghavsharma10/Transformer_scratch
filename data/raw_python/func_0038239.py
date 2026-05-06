def attribute_rewrite_map(self):
        """
        Example: long_name -> a_b

        :return: the rewrite map
        :rtype: dict
        """

        rewrite_map = dict()
        token_rewrite_map = self.generate_attribute_token_rewrite_map()

        for attribute_name, type_instance in self.getmembers():

            if isinstance(type_instance, DataType):
                attribute_tokens = attribute_name.split('_')

                rewritten_attribute_name = ''
                for token in attribute_tokens:
                    rewritten_attribute_name += token_rewrite_map[token] + "_"
                # remove the trailing underscore
                rewritten_attribute_name = rewritten_attribute_name[:-1]

                rewrite_map[attribute_name] = rewritten_attribute_name

        return rewrite_map