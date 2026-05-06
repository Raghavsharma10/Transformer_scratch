def model(model_names):
    """
        Creates the example directory structure necessary for a model service.
    """
    # for each model name we need to create
    for model_name in model_names:
        # the template context
        context = {
            'name': model_name,
        }

        # render the model template
        render_template(template='common', context=context)
        render_template(template='model', context=context)