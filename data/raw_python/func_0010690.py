def _serialize_data(self, my_dict):
        """
        Serialize a Dictionary into JSON
        """
        new_dict = {}
        for item in my_dict:
            if isinstance(my_dict[item], datetime):
                new_dict[item] = my_dict[item].strftime('%Y-%m-%d%H:%M:%S')
            else:
                new_dict[item] = str(my_dict[item])

        return json.dumps(new_dict)