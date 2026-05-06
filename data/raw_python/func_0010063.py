def get_attribute_value(self, name, index):
        """
        Returns the value associated with the given value index
        of the attribute with the given name.
        
        This is only applicable for nominal and string types.
        """
        if index == MISSING:
            return
        elif self.attribute_types[name] in NUMERIC_TYPES:
            at = self.attribute_types[name]
            if at == TYPE_INTEGER:
                return int(index)
            return Decimal(str(index))
        else:
            assert self.attribute_types[name] == TYPE_NOMINAL
            cls_index, cls_value = index.split(':')
            #return self.attribute_data[name][index-1]
            if cls_value != MISSING:
                assert cls_value in self.attribute_data[name], \
                    'Predicted value "%s" but only values %s are allowed.' \
                        % (cls_value, ', '.join(self.attribute_data[name]))
            return cls_value