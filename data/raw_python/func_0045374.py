def bind_context(context_filename):
    """
    loads context from file and binds to it

    :param: context_filename absolute path of the context file

    called by featuredjango.startup.select_product
    prior to selecting the individual features
    """

    global PRODUCT_CONTEXT
    if PRODUCT_CONTEXT is None:
        with open(context_filename) as contextfile:
            try:
                context = json.loads(contextfile.read())
            except ValueError as e:
                raise ContextParseError('Error parsing %s: %s' % (context_filename, str(e)))
            context['PRODUCT_CONTEXT_FILENAME'] = context_filename
            context['PRODUCT_EQUATION_FILENAME'] = os.environ['PRODUCT_EQUATION_FILENAME']
            context['PRODUCT_NAME'] = os.environ['PRODUCT_NAME']
            context['CONTAINER_NAME'] = os.environ['CONTAINER_NAME']
            context['PRODUCT_DIR'] = os.environ['PRODUCT_DIR']
            context['CONTAINER_DIR'] = os.environ['CONTAINER_DIR']
            context['APE_ROOT_DIR'] = os.environ['APE_ROOT_DIR']
            context['APE_GLOBAL_DIR'] = os.environ['APE_GLOBAL_DIR']
            PRODUCT_CONTEXT = ContextAccessor(context)
    else:
        # bind_context called but context already bound
        # harmless rebind (with same file) is ignored
        # otherwise this is a serious error
        if PRODUCT_CONTEXT.PRODUCT_CONTEXT_FILENAME != context_filename:
            raise ContextBindingError('product context bound multiple times using different data!')