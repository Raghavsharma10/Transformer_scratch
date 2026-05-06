def return_selected_form_items(form_info):
        """
        It returns chosen keys list from a given form.

        Args:
            form_info: serialized list of dict form data
        Returns:
            selected_keys(list): Chosen keys list
            selected_names(list): Chosen channels' or subscribers' names.
        """
        selected_keys = []
        selected_names = []
        for chosen in form_info:
            if chosen['choice']:
                selected_keys.append(chosen['key'])
                selected_names.append(chosen['name'])

        return selected_keys, selected_names