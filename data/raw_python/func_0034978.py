def constructFiniteStateMachine(inputs, outputs, states, table, initial,
                                richInputs, inputContext, world,
                                logger=LOGGER):
    """
    Construct a new finite state machine from a definition of its states.

    @param inputs: Definitions of all input symbols the resulting machine will
        need to handle, as a L{twisted.python.constants.Names} subclass.

    @param outputs: Definitions of all output symbols the resulting machine is
        allowed to emit, as a L{twisted.python.constants.Names} subclass.

    @param states: Definitions of all possible states the resulting machine
        will be capable of inhabiting, as a L{twisted.python.constants.Names}
        subclass.

    @param table: The state transition table, defining which output and next
        state results from the receipt of any and all inputs in any and all
        states.
    @type table: L{TransitionTable}

    @param initial: The state the machine will start in (one of the symbols
        from C{states}).

    @param richInputs: A L{list} of types which correspond to each of the input
        symbols from C{inputs}.
    @type richInputs: L{list} of L{IRichInput} I{providers}

    @param inputContext: A L{dict} mapping output symbols to L{Interface}
        subclasses describing the requirements of the inputs which lead to
        them.

    @param world: An object responsible for turning FSM outputs into observable
        side-effects.
    @type world: L{IOutputExecutor} provider

    @param logger: The logger to which to write messages.
    @type logger: L{eliot.ILogger} or L{NoneType} if there is no logger.

    @return: An L{IFiniteStateMachine} provider
    """
    table = table.table

    _missingExtraCheck(
        set(table.keys()), set(states.iterconstants()),
        ExtraTransitionState, MissingTransitionState)

    _missingExtraCheck(
        set(i for s in table.values() for i in s), set(inputs.iterconstants()),
        ExtraTransitionInput, MissingTransitionInput)

    _missingExtraCheck(
        set(output for s in table.values() for transition in s.values() for output in transition.output),
        set(outputs.iterconstants()),
        ExtraTransitionOutput, MissingTransitionOutput)

    try:
        _missingExtraCheck(
            set(transition.nextState for s in table.values() for transition in s.values()),
            set(states.iterconstants()),
            ExtraTransitionNextState, MissingTransitionNextState)
    except MissingTransitionNextState as e:
        if e.args != ({initial},):
            raise

    if initial not in states.iterconstants():
        raise InvalidInitialState(initial)

    extraInputContext = set(inputContext) - set(outputs.iterconstants())
    if extraInputContext:
        raise ExtraInputContext(extraInputContext)

    _checkConsistency(richInputs, table, inputContext)

    fsm = _FiniteStateMachine(inputs, outputs, states, table, initial)
    executor = IOutputExecutor(world)
    interpreter = _FiniteStateInterpreter(
        tuple(richInputs), inputContext, fsm, executor)
    if logger is not None:
        interpreter = FiniteStateLogger(
            interpreter, logger, executor.identifier())
    return interpreter