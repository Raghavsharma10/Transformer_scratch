def get_elements(self, object_list):
        """
        Recursive method to iterate the tree of children in order to flatten it

        :param object_list:
        :return:
        """
        result = []
        for item in object_list:
            if isinstance(item, list):
                result += self.get_elements(item)
            elif isinstance(item, TranslatableModel):
                result.append(item)
        return result