def get_filled_structure(self, subgroup=None):
        '''
        method in charged of filling an structure containing the object fields
        values taking into account the 'group' attribute from the corresponding
        form object, which is necesary to fill the details form as it is configured
        in the 'group' attribute
        '''
        # initilize the result structure
        result = []

        # the object corresponding model content is taken into a dictionary
        object_content = model_to_dict(self.object)

        # generallically some common or specific fields are not interesting
        if 'exclude_fields' not in dir(self):
            self.exclude_fields = []

        self.exclude_fields.append("id")

        for field in (self.exclude_fields):
            if field in object_content.keys():
                object_content.pop(field)
        # following is going to be created an structure with the appropieate caption
        # for every existing field in the current model
        verbose_names = {}
        for field in object_content.keys():
            verbose_names[field] = self.model._meta.get_field(field).verbose_name
        # the found fields in the groups structure are going to be taked into account
        gr_object_content = []

        if subgroup:
            group_array = subgroup
        else:
            group_array = self.groups

        for group in group_array:

            # raise Exception(group)
            item = {}

            item["name"] = smart_text(group[0])
            item["col"] = group[1]
            item_elements = group[2:]

            sublist = []

            idx = 0
            for item_element in item_elements:
                # the element can contains another groups
                if (idx > 1) and (type(item_element) == tuple):
                    # Recursive
                    sublist.append(self.get_filled_structure([subgroup]))
                else:
                    filter_field = None
                    # Check if it is a list
                    if type(item_element) == list:
                        # if it is a list, that means that can be found the
                        # corresponding values for colums and any other
                        field = item_element[0]
                        # take into account that field caption can be passed as
                        # third list element
                        if len(item_element) >= 3 and item_element[2]:
                            verbose_names[field] = _(item_element[2])
                        if len(item_element) >= 9:
                            filter_field = item_element[8]
                    else:

                        field = item_element

                    if field not in verbose_names:
                        if field.startswith('get_') and field.endswith('_display'):
                            label_field = remove_getdisplay(field)
                            if self.model:
                                try:
                                    verbose_names[field] = self.model._meta.get_field(label_field).verbose_name
                                except FieldDoesNotExist:
                                    verbose_names[field] = _(label_field)
                            else:
                                verbose_names[field] = _(label_field)
                        else:
                            label_field = field
                            verbose_names[field] = _(label_field)

                    args = {}

                    value = None
                    for field_split in field.split('__'):
                        if value is None:
                            try:
                                verbose_names[field] = self.object._meta.get_field(field_split).verbose_name
                            except AttributeError:
                                pass
                            except FieldDoesNotExist:
                                pass

                            value = getattr(self.object, field_split, None)
                        else:
                            try:
                                verbose_names[field] = value._meta.get_field(field_split).verbose_name
                            except AttributeError:
                                pass
                            except FieldDoesNotExist:
                                pass
                            value = getattr(value, field_split, None)

                    if callable(value):
                        # if 'request' in value.func_code.co_varnames:
                        related = (getattr(value, 'all', None) is not None)
                        if related:
                            value = ", ".join([str(x) for x in value.all()])
                        else:
                            if 'request' in value.__code__.co_varnames:
                                args['request'] = self.request
                                # Call the method
                            value = value(**args)

                    sublist.append({
                        "name": _(verbose_names[field]),
                        "value": value,
                        "filter": filter_field,
                    })
                    gr_object_content.append(field)

                # Increment index
                idx += 1

            item["value"] = sublist
            result.append(item)

        for field in object_content.keys():
            item = {}
            if field not in gr_object_content:
                item["name"] = _(verbose_names[field])
                item["value"] = getattr(self.object, field)
                result.append(item)

        return result