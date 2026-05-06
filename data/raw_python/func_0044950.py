def generate_context(force_overwrite=False, drop_secret_key=False):
    """
    Generates context.json
    """

    print('... generating context')
    context_fp = '%s/context.json' % os.environ['PRODUCT_DIR']
    context = {}

    if os.path.isfile(context_fp):
        print('... augment existing context.json')

        with open(context_fp, 'r') as context_f:
            content = context_f.read().strip() or '{}'
            try:
                context = json.loads(content)
            except ValueError:
                print('ERROR: not valid json in your existing context.json!!!')
                return

        if force_overwrite:
            print('... overwriting existing context.json')
            if drop_secret_key:
                print('... generating new SECRET_KEY')
                context = {}
            else:
                print('... using existing SECRET_KEY from existing context.json')
                context = {'SECRET_KEY': context['SECRET_KEY']}

    with open(context_fp, 'w') as context_f:
        new_context = tasks.get_context_template()

        new_context.update(context)
        context_f.write(json.dumps(new_context, indent=4, sort_keys=True))
    print()
    print('*** Successfully generated context.json')