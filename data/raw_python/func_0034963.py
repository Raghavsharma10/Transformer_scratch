def _compute_stacksize(self):
        '''
        Given this object's code list, compute its maximal stack usage.
        This is done by scanning the code, and computing for each opcode
        the stack state at the opcode.

        '''

        # get local access to code, save some attribute lookups later
        code = self.code

        # A mapping from labels to their positions in the code list
        label_pos = { op : pos
                        for pos, (op, arg) in enumerate(code)
                        if isinstance(op, Label)
                    }

        # sf_targets are the targets of SETUP_FINALLY opcodes. They are
        # recorded because they have special stack behaviour. If an exception
        # was raised in the block pushed by a SETUP_FINALLY opcode, the block
        # is popped and 3 objects are pushed. On return or continue, the
        # block is popped and 2 objects are pushed. If nothing happened, the
        # block is popped by a POP_BLOCK opcode and 1 object is pushed by a
        # (LOAD_CONST, None) operation.
        #
        # In Python 3, the targets of SETUP_WITH have similar behavior,
        # complicated by the fact that they also have an __exit__ method
        # stacked and what it returns determines what they pop. So their
        # stack depth is one greater, a fact we are going to ignore for the
        # time being :-/
        #
        # Our solution is to record the stack state of SETUP_FINALLY targets
        # as having 3 objects pushed, which is the maximum. However, to make
        # stack recording consistent, the get_next_stacks function will always
        # yield the stack state of the target as if 1 object was pushed, but
        # this will be corrected in the actual stack recording.

        sf_targets = set( label_pos[arg]
                          for op, arg in code
                          if op == SETUP_FINALLY or op == SETUP_WITH
                        )

        # What we compute - for each opcode, its stack state, as an n-tuple.
        # n is the number of blocks pushed. For each block, we record the number
        # of objects pushed.
        stacks = [None] * len(code)

        def get_next_stacks(pos, curstack):
            """
            Get a code position and the stack state before the operation
            was done, and yield pairs (pos, curstack) for the next positions
            to be explored - those are the positions to which you can get
            from the given (pos, curstack).

            If the given position was already explored, nothing will be yielded.
            """
            op, arg = code[pos]

            if isinstance(op, Label):
                # We should check if we already reached a node only if it is
                # a label.

                if pos in sf_targets:
                    # Adjust a SETUP_FINALLY from 1 to 3 stack entries.
                    curstack = curstack[:-1] + (curstack[-1] + 2,)

                if stacks[pos] is None:
                    stacks[pos] = curstack
                else:
                    if stacks[pos] != curstack:
                        raise ValueError("Inconsistent code")
                    return

            def newstack(n):
                # Return a new stack, modified by adding n elements to the last
                # block
                if curstack[-1] + n < 0:
                    raise ValueError("Popped a non-existing element")
                return curstack[:-1] + (curstack[-1]+n,)

            if not isopcode(op):
                # label or SetLineno - just continue to next line
                yield pos+1, curstack

            elif op in ( RETURN_VALUE, RAISE_VARARGS ):
                # No place in particular to continue to
                pass

            elif op in (JUMP_FORWARD, JUMP_ABSOLUTE):
                # One possibility for a jump
                yield label_pos[arg], curstack

            elif op in (POP_JUMP_IF_FALSE, POP_JUMP_IF_TRUE):
                # Two possibilities for a jump
                yield label_pos[arg], newstack(-1)
                yield pos+1, newstack(-1)

            elif op in (JUMP_IF_TRUE_OR_POP, JUMP_IF_FALSE_OR_POP):
                # Two possibilities for a jump
                yield label_pos[arg], curstack
                yield pos+1, newstack(-1)

            elif op == FOR_ITER:
                # FOR_ITER pushes next(TOS) on success, and pops TOS and jumps
                # on failure
                yield label_pos[arg], newstack(-1)
                yield pos+1, newstack(1)

            elif op == BREAK_LOOP:
                # BREAK_LOOP goes to the end of a loop and pops a block
                # but like RETURN_VALUE we have no instruction position
                # to give. For now treat like RETURN_VALUE
                pass

            elif op == CONTINUE_LOOP:
                # CONTINUE_LOOP jumps to the beginning of a loop which should
                # already have been discovered. It does not change the stack
                # state nor does it create or pop a block.
                #yield label_pos[arg], curstack
                #yield label_pos[arg], curstack[:-1]
                pass

            elif op == SETUP_LOOP:
                # We continue with a new block.
                # On break, we jump to the label and return to current stack
                # state.
                yield label_pos[arg], curstack
                yield pos+1, curstack + (0,)

            elif op == SETUP_EXCEPT:
                # We continue with a new block.
                # On exception, we jump to the label with 3 extra objects on
                # stack
                yield label_pos[arg], newstack(3)
                yield pos+1, curstack + (0,)

            elif op == SETUP_FINALLY or op == SETUP_WITH :
                # We continue with a new block.
                # On exception, we jump to the label with 3 extra objects on
                # stack, but to keep stack recording consistent, we behave as
                # if we add only 1 object. Extra 2 will be added to the actual
                # recording.
                yield label_pos[arg], newstack(1)
                yield pos+1, curstack + ( int(op == SETUP_WITH) ,)

            elif op == POP_BLOCK:
                # Just pop the block
                yield pos+1, curstack[:-1]

            elif op == END_FINALLY :
                # Since stack recording of SETUP_FINALLY targets is of 3 pushed
                # objects (as when an exception is raised), we pop 3 objects.
                yield pos+1, newstack(-3)

            elif op == _WITH_CLEANUP_OPCODE:
                # Since WITH_CLEANUP[_START] is always found after SETUP_FINALLY
                # targets, and the stack recording is that of a raised
                # exception, we can simply pop 1 object and let END_FINALLY
                # pop the remaining 3.
                yield pos+1, newstack(-1)

            else:
                # nothing special, use the CPython value
                yield pos+1, newstack( stack_effect( op, arg ) )


        # Now comes the calculation: open_positions holds positions which are
        # yet to be explored. In each step we take one open position, and
        # explore it by appending the positions to which it can go, to
        # open_positions. On the way, we update maxsize.
        #
        # open_positions is a list of tuples: (pos, stack state)
        #
        # Sneaky Python coding trick here. get_next_stacks() is a generator,
        # it contains yield statements. So when we call get_next_stacks()
        # what is returned is an iterator. However, the yield statements in
        # get_next_stacks() are not in a loop as usual; rather it is
        # straight-line code that will execute 0, 1 or 2 yields depending on
        # the Opcode at pos.
        #
        # the list.extend() method takes an iterator and exhausts it, adding
        # all yielded values to the list. Hence the statement
        #
        #   open_positions.extend(get_next_stacks(pos,curstack))
        #
        # appends 0, 1 or 2 tuples (pos, stack_state) to open_positions.

        maxsize = 0
        open_positions = [(0, (0,))]
        while open_positions:
            pos, curstack = open_positions.pop()
            maxsize = max(maxsize, sum(curstack))
            open_positions.extend(get_next_stacks(pos, curstack))

        return maxsize