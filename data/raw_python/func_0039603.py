def default_profiler(f, _type, _value):
    ''' inspects an input frame and pretty prints the following:

        <src-path>:<src-line> -> <function-name>
        <source-code>
        <local-variables>
        ----------------------------------------
    '''
    try:
        profile_print(
            '\n'.join([
                get_frame_src(f),
                get_locals(f),
                '----------------------------------------'
            ])
        )
    except:
        pass