def add_field_processors(config, processors, model, field):
    """ Add processors for model field.

    Under the hood, regular nefertari event subscribed is created which
    calls field processors in order passed to this function.

    Processors are passed following params:

    * **new_value**: New value of of field.
    * **instance**: Instance affected by request. Is None when set of
      items is updated in bulk and when item is created.
    * **field**: Instance of nefertari.utils.data.FieldData instance
      containing data of changed field.
    * **request**: Current Pyramid Request instance.
    * **model**: Model class affected by request.
    * **event**: Underlying event object.

    Each processor must return processed value which is passed to next
    processor.

    :param config: Pyramid Congurator instance.
    :param processors: Sequence of processor functions.
    :param model: Model class for field if which processors are
        registered.
    :param field: Field name for which processors are registered.
    """
    before_change_events = (
        BeforeCreate,
        BeforeUpdate,
        BeforeReplace,
        BeforeUpdateMany,
        BeforeRegister,
    )

    def wrapper(event, _processors=processors, _field=field):
        proc_kw = {
            'new_value': event.field.new_value,
            'instance': event.instance,
            'field': event.field,
            'request': event.view.request,
            'model': event.model,
            'event': event,
        }
        for proc_func in _processors:
            proc_kw['new_value'] = proc_func(**proc_kw)

        event.field.new_value = proc_kw['new_value']
        event.set_field_value(_field, proc_kw['new_value'])

    for evt in before_change_events:
        config.add_subscriber(wrapper, evt, model=model, field=field)