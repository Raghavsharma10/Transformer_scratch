def enhance(self):
        """ Function enhance
        Enhance the object with new item or enhanced items
        """
        if self.objName in ['hosts', 'hostgroups',
                            'puppet_classes']:
            from foreman.itemSmartClassParameter\
                import ItemSmartClassParameter
            self.update({'smart_class_parameters':
                        SubDict(self.api, self.objName,
                                self.payloadObj, self.key,
                                ItemSmartClassParameter)})