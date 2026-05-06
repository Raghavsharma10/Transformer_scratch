def transform(source):
    '''Used to convert the source code, making use of known transformers.

       "transformers" are modules which must contain a function

           transform_source(source)

       which returns a tranformed source.
       Some transformers (for example, those found in the standard library
       module lib2to3) cannot cope with non-standard syntax; as a result, they
       may fail during a first attempt. We keep track of all failing
       transformers and keep retrying them until either they all succeeded
       or a fixed set of them fails twice in a row.
    '''
    source = extract_transformers_from_source(source)

    # Some transformer fail when multiple non-Python constructs
    # are present. So, we loop multiple times keeping track of
    # which transformations have been unsuccessfully performed.
    not_done = transformers
    while True:
        failed = {}
        for name in not_done:
            tr_module = import_transformer(name)
            try:
                source = tr_module.transform_source(source)
            except Exception as e:
                failed[name] = tr_module
                # from traceback import print_exc
                # print("Unexpected exception in transforms.transform",
                #       e.__class__.__name__)
                # print_exc()

        if not failed:
            break
        # Insanity is doing the same Tting over and overaAgain and
        # expecting different results ...
        # If the exact same set of transformations are not performed
        # twice in a row, there is no point in trying out a third time.
        if failed == not_done:
            print("Warning: the following transforms could not be done:")
            for key in failed:
                print(key)
            break
        not_done = failed  # attempt another pass

    return source