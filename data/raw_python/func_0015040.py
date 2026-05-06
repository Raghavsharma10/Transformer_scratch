def _get_field_python_type(model, name):
        """
        Gets the python type for the attribute on the model
        with the name provided.

        :param Model model: The SqlAlchemy model class.
        :param unicode name: The column name on the model
            that you are attempting to get the python type.
        :return: The python type of the column
        :rtype: type
        """
        try:
            return getattr(model, name).property.columns[0].type.python_type
        except AttributeError:  # It's a relationship
            parts = name.split('.')
            model = getattr(model, parts.pop(0)).comparator.mapper.class_
            return AlchemyManager._get_field_python_type(model, '.'.join(parts))
        except NotImplementedError:
            # This is for pickle type columns.
            return object