def build_nested_field(self, field_name, relation_info, nested_depth):
        """ Use PriceEstimateSerializer to serialize estimate children """
        if field_name != 'children':
            return super(PriceEstimateSerializer, self).build_nested_field(field_name, relation_info, nested_depth)
        field_class = self.__class__
        field_kwargs = {'read_only': True, 'many': True, 'context': {'depth': nested_depth - 1}}
        return field_class, field_kwargs