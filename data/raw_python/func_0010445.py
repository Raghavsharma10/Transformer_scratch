def get_field_names(self):
        """
        Different field names depending on self.add setting (see load_data)
        For BaseIO
        """
        if self.add:
            return ['date', 'elem', 'value'] + [flag for flag in self.add]
        else:
            field_names = ['date']
            for elem in self.parameter:
                # namedtuple doesn't like numeric field names
                if elem.isdigit():
                    elem = "e%s" % elem
                field_names.append(elem)
            return field_names