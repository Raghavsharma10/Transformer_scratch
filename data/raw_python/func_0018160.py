def items_for_result(view, result):
    """
    Generates the actual list of data.
    """
    model_admin = view.model_admin
    for field_name in view.list_display:
        empty_value_display = model_admin.get_empty_value_display()
        row_classes = ['field-%s' % field_name]
        try:
            f, attr, value = lookup_field(field_name, result, model_admin)
        except ObjectDoesNotExist:
            result_repr = empty_value_display
        else:
            empty_value_display = getattr(attr, 'empty_value_display', empty_value_display)
            if f is None or f.auto_created:
                allow_tags = getattr(attr, 'allow_tags', False)
                boolean = getattr(attr, 'boolean', False)
                if boolean or not value:
                    allow_tags = True

                if django.VERSION >= (1, 9):
                    result_repr = display_for_value(value, empty_value_display, boolean)
                else:
                    result_repr = display_for_value(value, boolean)
                # Strip HTML tags in the resulting text, except if the
                # function has an "allow_tags" attribute set to True.
                if allow_tags:
                    result_repr = mark_safe(result_repr)
                if isinstance(value, (datetime.date, datetime.time)):
                    row_classes.append('nowrap')
            else:
                if isinstance(f, models.ManyToOneRel):
                    field_val = getattr(result, f.name)
                    if field_val is None:
                        result_repr = empty_value_display
                    else:
                        result_repr = field_val
                else:
                    if django.VERSION >= (1, 9):
                        result_repr = display_for_field(value, f, empty_value_display)
                    else:
                        result_repr = display_for_field(value, f)

                if isinstance(f, (models.DateField, models.TimeField, models.ForeignKey)):
                    row_classes.append('nowrap')
        if force_text(result_repr) == '':
            result_repr = mark_safe('&nbsp;')
        row_classes.extend(model_admin.get_extra_class_names_for_field_col(field_name, result))
        row_attributes_dict = model_admin.get_extra_attrs_for_field_col(field_name, result)
        row_attributes_dict['class'] = ' ' . join(row_classes)
        row_attributes = ''.join(' %s="%s"' % (key, val) for key, val in row_attributes_dict.items())
        row_attributes_safe = mark_safe(row_attributes)
        yield format_html('<td{}>{}</td>', row_attributes_safe, result_repr)