def simple_input(self, variables):
        """
        Use this method to get simple input as python object, with all
        templates filled in

        :param variables:
        :return: python object

        """
        json_args = fill_template_str(json.dumps(self.data), variables)
        return try_get_objects(json_args)