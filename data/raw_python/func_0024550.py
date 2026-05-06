def get_object_query_dict(self):
        """returns objects keys according to self.object_query_code
        which can be json encoded queryset filter dict or key=value set
         in the following format: ```"key=val, key2 = val2 , key3= value with spaces"```

        Returns:
             (dict): Queryset filtering dicqt
        """
        if isinstance(self.object_query_code, dict):
            # _DATE_ _DATETIME_
            return self.object_query_code
        else:
            # comma separated, key=value pairs. wrapping spaces will be ignored
            # eg: "key=val, key2 = val2 , key3= value with spaces"
            return dict(pair.split('=') for pair in self.object_query_code.split(','))