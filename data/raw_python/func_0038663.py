def init_target_object(self, target, data):
        """
        Initializes target object and assign extra objects to target as attributes
        """
        target_object = self.init_single_object(target, data.pop(target, data))
        for key, item in data.items():
            key_alias = MANAGER_ALIASES.get(key, key)
            if item:
                # Item is an OrderedDict with 4 possible structure patterns:
                #   - Just an OrderedDict with (key - value)'s
                #   - OrderedDict with single (key - OrderedDict)
                #   - OrderedDict with multiple (key - OrderedDict)'s
                #   - String (like CreativeCode model)
                if isinstance(item, str):
                    children = item
                else:
                    first_key = list(item.keys())[0]
                    if isinstance(item[first_key], OrderedDict):
                        instances = item.values()
                        if len(instances) > 1:
                            children = [self.init_single_object(key_alias, instance) for instance in instances]
                        else:
                            children = self.init_single_object(key_alias, list(instances)[0])
                    else:
                        children = self.init_single_object(key_alias, item)
                setattr(target_object, key.lower(), children)
            else:
                setattr(target_object, key.lower(), None)
        return target_object