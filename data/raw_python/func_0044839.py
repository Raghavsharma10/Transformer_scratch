def render_field(field):
    """
    渲染字段验证代码
    :param field:
     :type field: django.forms.Field
    :return:
    """
    field = field.field if isinstance(field, forms.BoundField) else field
    validators = {}

    def no_compare_validator():
        return not ('lessThan' in validators or 'greaterThan' in validators or 'between' in validators)

    if field.required:
        validators['notEmpty'] = {}
    validator_codes = [item.code for item in field.validators]
    for v in field.validators:
        if isinstance(v, MinLengthValidator):
            vc = validators.get('stringLength', {})
            vc['min'] = field.min_length
            validators.update({'stringLength': vc})
        elif isinstance(v, MaxLengthValidator):
            vc = validators.get('stringLength', {})
            vc['max'] = field.max_length
            validators.update({'stringLength': vc})
        elif isinstance(v, (MinValueValidator, MaxValueValidator)):
            if 'min_value' in validator_codes and 'max_value' in validator_codes:
                vc = validators.get('between', {})
                if v.code == 'min_value':
                    vc['min'] = field.min_value
                else:
                    vc['max'] = field.max_value
                validators.update({'between': vc})
            elif v.code == 'min_value':
                validators['greaterThan'] = {'value': field.min_value}
            elif v.code == 'max_value':
                validators['lessThan'] = {'value': field.max_value}
        elif isinstance(v, BaseBV):
            validators.update(v.get_validator_code())

    if isinstance(field, (fields.DecimalField, fields.FloatField)) and no_compare_validator():
        validators['numeric'] = {}
    elif isinstance(field, fields.IntegerField) and no_compare_validator():
        validators['integer'] = {}
    elif isinstance(field, (fields.DateField, fields.DateTimeField)):
        formats = field.input_formats
        if formats:
            validators['date'] = {'format': convert_datetime_python_to_javascript(formats[0])}
    elif isinstance(field, fields.TimeField):
        validators['regexp'] = {'regexp': '^((([0-1]?[0-9])|([2][0-3])):)(([0-5][0-9]):)([0-5][0-9])$',
        }
    elif isinstance(field, fields.URLField):
        validators['uri'] = {}
    elif isinstance(field, fields.EmailField):
        validators['emailAddress'] = {}
    elif isinstance(field, fields.ImageField):
        if 'file' not in validators:
            validators.update(ImageFileValidator().get_validator_code())
    return {'validators': validators}