def build(cls, data, *args, **kwargs):
        """
        Constructs a classification or regression tree in a single batch by
        analyzing the given data.
        """
        assert isinstance(data, Data)
        if data.is_continuous_class:
            fitness_func = gain_variance
        else:
            fitness_func = get_gain
        
        t = cls(data=data, *args, **kwargs)
        t._data = data
        t.sample_count = len(data)
        t._tree = create_decision_tree(
            data=data,
            attributes=data.attribute_names,
            class_attr=data.class_attribute_name,
            fitness_func=fitness_func,
            wrapper=t,
        )
        return t