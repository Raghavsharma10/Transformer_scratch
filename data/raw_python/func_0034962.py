def from_code(cls, code_object):
        """
        Disassemble a Python code object and make a Code object from the bits.
        This is the expected way to make a Code instance. But you are welcome
        to call Code() directly if you wish.
        """
        # It's an annoyance to keep having to add ".__code__" to a function
        # name, so let's automate that when needed.
        if isinstance( code_object, types.FunctionType ) :
            code_object = code_object.__code__

        # get the actual bytecode string out of the code object
        co_code = code_object.co_code

        # Use dis.findlabels to locate the labeled bytecodes, that is, the
        # ones that are jump targets. (They are "labeled" in a disassembly
        # printout.) Store the list as a dict{ addr: Label object} for easy
        # lookup.

        labels = dict((addr, Label()) for addr in findlabels(co_code))

        # Make a dict{ source_line : offset } for the source lines in the code.

        linestarts = dict(cls._findlinestarts(code_object))

        cellfree = code_object.co_cellvars + code_object.co_freevars

        # Create a CodeList object to represent the bytecode string.

        code = CodeList()   # receives (op,arg) tuples
        n = len(co_code)    # number bytes in the bytecode string
        i = 0               # index over the bytecode string
        extended_arg = 0    # upper 16 bits of an extended arg

        # Iterate over the bytecode string expanding it into (Opcode,arg) tuples.

        while i < n:
            # First byte is the opcode
            op = Opcode( co_code[i] )

            # If this op is a jump-target, insert (Label,) ahead of it.
            if i in labels:
                code.append((labels[i], None))

            # If this op is the first from a source line, insert
            # (SetLineno, line#) ahead of it.
            if i in linestarts:
                code.append((SetLineno, linestarts[i]))

            i += 1 # step index to the argument if any

            if op not in hasargx :
                # No argument, push the minimal tuple, done.
                code.append((op, None))
            else:
                # op takes an argument. Look for MAKE_FUNCTION or MAKE_CLOSURE.
                if op in hascode :
                    # special case: with these opcodes, at runtime, TOS1 should
                    # be a code object. We require the normal opcode sequence:
                    #    LOAD_CONST the code object
                    #    LOAD_CONST the name of the function
                    #    MAKE_FUNCTION/CLOSURE
                    # When this exists, go back and convert the argument of the
                    # first LOAD_CONST from a code object to a Code object.
                    if len(code) >= 2 \
                       and code[-2][0] == LOAD_CONST \
                       and code[-1][0] == LOAD_CONST \
                       and isinstance( code[-2][1], types.CodeType ) :
                        code[-2] = ( Opcode(LOAD_CONST), Code.from_code( code[-2][1] ) )
                    else :
                        raise ValueError(
                            'Invalid opcode sequence for MAKE_FUNCTION/MAKE_CLOSURE'
                        )
                    # now continue and handle the argument of MAKE_F/C normally.

                # Assemble the argument value from two bytes plus an extended
                # arg when present.
                arg = co_code[i] + co_code[i+1]*256 + extended_arg
                extended_arg = 0 # clear extended arg bits if any
                i += 2 # Step over the argument

                if op == opcode.EXTENDED_ARG:
                    # The EXTENDED_ARG op is just a way of storing the upper
                    # 16 bits of a 32-bit arg in the bytestream. Collect
                    # those bits, but generate no code tuple.
                    extended_arg = arg << 16

                elif op in hasconst:
                    # When the argument is a constant, put the constant
                    # itself in the opcode tuple. If that constant is a code
                    # object, the test above (if op in hascode) will later
                    # convert it into a Code object.
                    code.append((op, code_object.co_consts[arg]))

                elif op in hasname:
                    # When the argument is a name, put the name string itself
                    # in the opcode tuple.
                    code.append((op, code_object.co_names[arg]))

                elif op in hasjabs:
                    # When the argument is an absolute jump, put the label
                    # in the tuple (in place of the label list index)
                    code.append((op, labels[arg]))

                elif op in hasjrel:
                    # When the argument is a relative jump, put the label
                    # in the tuple in place of the forward offset.
                    code.append((op, labels[i + arg]))

                elif op in haslocal:
                    # When the argument is a local var, put the name string
                    # in the tuple.
                    code.append((op, code_object.co_varnames[arg]))

                elif op in hascompare:
                    # When the argument is a relation (like ">=") put that
                    # string in the tuple instead.
                    code.append((op, cmp_op[arg]))

                elif op in hasfree:
                    code.append((op, cellfree[arg]))

                else:
                    # whatever, just put the arg in the tuple
                    code.append((op, arg))

        # Store certain flags from the code object as booleans for convenient
        # reference as Code members.

        varargs = bool(code_object.co_flags & CO_VARARGS)
        varkwargs = bool(code_object.co_flags & CO_VARKEYWORDS)
        newlocals = bool(code_object.co_flags & CO_NEWLOCALS)

        # Get the names of arguments as strings, from the varnames tuple. The
        # order of name strings in co_varnames is:
        #   co_argcount names of regular (positional-or-keyword) arguments
        #   names of co_kwonlyargcount keyword-only arguments if any
        #   name of a *vararg argument
        #   name of a **kwarg argument if any (not present if kwonlyargs > 0)
        #   names of other local variables
        # Hence the count of argument names is
        #   co_argcount + co_kwonlyargcount + varargs + varkwargs
        nargs = code_object.co_argcount + code_object.co_kwonlyargcount + varargs + varkwargs
        args = code_object.co_varnames[ : nargs ]

        # Preserve a docstring if any. If there are constants and the first
        # constant is a string, Python assumes that's a docstring.
        docstring = None
        if code_object.co_consts and isinstance(code_object.co_consts[0], str):
            docstring = code_object.co_consts[0]

        # Funnel all the collected bits through the Code.__init__() method.
        return cls( code = code,
                    freevars = code_object.co_freevars,
                    args = args,
                    varargs = varargs,
                    varkwargs = varkwargs,
                    kwonlyargcount = code_object.co_kwonlyargcount,
                    newlocals = newlocals,
                    coflags = code_object.co_flags,
                    name = code_object.co_name,
                    filename = code_object.co_filename,
                    firstlineno = code_object.co_firstlineno,
                    docstring = docstring
                    )