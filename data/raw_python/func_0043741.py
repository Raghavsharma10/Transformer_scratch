def tabulate(json_model):
    
    '''
        a function to add the tabulate method to a jsonModel object
        
    :param json_model: jsonModel object
    :return: jsonModel object
    '''

    import types
    from jsonmodel._extensions import tabulate as _tabulate
    try:
        from tabulate import tabulate
    except:
        import sys
        print('jsonmodel.extensions.tabulate requires the tabulate module. try: pip install tabulate')
        sys.exit(1)

    setattr(json_model, 'tabulate', _tabulate.__get__(json_model, types.MethodType))

    return json_model