def translated(structure, values, lang_spec):
    """Return code associated to given structure and values, 
    translate with given language specification."""
    # LANGUAGE SPECS
    indentation = '\t'
    endline = '\n'


    object_code = ""
    stack = []
    # define shortcuts to behavior
    push = lambda x: stack.append(x)
    pop  = lambda  : stack.pop()
    last = lambda  : stack[-1] if len(stack) > 0 else ' '
    def indented_code(s, level, end):
        return lang_spec[INDENTATION]*level + s + end

    # recreate python structure, and replace type by value
    level = 0
    CONDITIONS = [LEXEM_TYPE_PREDICAT, LEXEM_TYPE_CONDITION]
    ACTION = LEXEM_TYPE_ACTION
    DOWNLEVEL = LEXEM_TYPE_DOWNLEVEL
    for lexem_type in structure:
        if lexem_type is ACTION:
            # place previous conditions if necessary
            if last() in CONDITIONS:
                # construct conditions lines
                value, values = values[0:len(stack)], values[len(stack):]
                object_code += (indented_code(lang_spec[BEG_CONDITION] 
                    + lang_spec[LOGICAL_AND].join(value) 
                    + lang_spec[END_CONDITION], 
                    level, 
                    lang_spec[END_LINE]
                ))
                # if provided, print the begin block token on a new line
                if len(lang_spec[BEG_BLOCK]) > 0:
                    object_code += indented_code( 
                        lang_spec[BEG_BLOCK],
                        level, 
                        lang_spec[END_LINE]
                    )
                stack = []
                level += 1
            # and place the action
            object_code += indented_code(
                lang_spec[BEG_ACTION] + values[0], 
                level, 
                lang_spec[END_ACTION]+lang_spec[END_LINE]
            )
            values = values[1:]
        elif lexem_type in CONDITIONS:
            push(lexem_type)
        elif lexem_type is DOWNLEVEL:
            if last() not in CONDITIONS:
                # down level, and add a END_BLOCK only if needed
                level -= 1
                if level >= 0:
                    object_code += indented_code(
                        lang_spec[END_BLOCK], level,
                        lang_spec[END_LINE]
                    )
                else:
                    level = 0

    # add END_BLOCK while needed for reach level 0
    while level > 0:
        level -= 1
        if level >= 0:
            object_code += indented_code(
                lang_spec[END_BLOCK], level,
                lang_spec[END_LINE]
            )
        else:
            level = 0
    # Finished !
    return object_code