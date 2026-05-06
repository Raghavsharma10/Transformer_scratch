def _findlinestarts(code_object):
        """
        Find the offsets in a byte code which are the start of source lines.

        Generate pairs (offset, lineno) as described in Python/compile.c.

        This is a modified version of dis.findlinestarts. This version allows
        multiple "line starts" with the same line number. (The dis version
        conditions its yield on a test "if lineno != lastlineno".)

        FYI: code.co_lnotab is a byte array with one pair of bytes for each
        effective source line number in the bytecode. An effective line is
        one that generates code: not blank or comment lines. The first actual
        line number, typically the number of the "def" statement, is in
        code.co_firstlineno.

        An even byte of co_lnotab is the offset to the bytecode generated
        from the next effective line number. The following odd byte is an
        increment on the previous line's number to the next line's number.
        Thus co_firstlineno+co_lnotab[1] is the first effective line's
        number, and co_lnotab[0] is the number of bytes it generated.

        Note that an effective line number generates code by definition,
        hence the even byte cannot be zero; and as line numbers are
        monotonically increasing, the odd byte cannot be zero either.

        But what, the curious reader might ask, does Python do if a source
        line generates more than 255 bytes of code? In that *highly* unlikely
        case compile.c generates multiple pairs of (255,0) until it has
        accounted for all the generated code, then a final pair of
        (offset%256, lineincr).

        Oh, but what, the curious reader asks, do they do if there is a gap
        of more than 255 between effective line numbers? It is not unheard of
        to find blocks of comments larger than 255 lines (like this one?).
        Then compile.c generates pairs of (0, 255) until it has accounted for
        the line number difference and a final pair of (offset,lineincr%256).

        Uh, but...? Yes, what now, annoying reader? Well, does the following
        code handle these special cases of (255,0) and (0,255) properly?
        It handles the (0,255) case correctly, because of the "if byte_incr"
        test which skips the yield() but increments lineno. It does not handle
        the case of (255,0) correctly; it will yield false pairs (255,0).
        Fortunately that will only arise e.g. when disassembling some
        "obfuscated" code where most newlines are replaced with semicolons.

        Oh, and yes, the to_code() method does properly handle generation
        of the (255,0) and (0,255) entries correctly.

        """
        # grab the even bytes as integer byte_increments:
        byte_increments = [c for c in code_object.co_lnotab[0::2]]
        # grab the odd bytes as integer line_increments:
        line_increments = [c for c in code_object.co_lnotab[1::2]]

        lineno = code_object.co_firstlineno
        addr = 0
        for byte_incr, line_incr in zip(byte_increments, line_increments):
            if byte_incr:
                yield (addr, lineno)
                addr += byte_incr
            lineno += line_incr
        yield (addr, lineno)