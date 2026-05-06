def convert_to_order_dict(map_list):
        """ convert mapping in list to ordered dict
        @param (list) map_list
            [
                {"a": 1},
                {"b": 2}
            ]
        @return (OrderDict)
            OrderDict({
                "a": 1,
                "b": 2
            })
        """
        ordered_dict = OrderedDict()
        for map_dict in map_list:
            ordered_dict.update(map_dict)
        
        return ordered_dict