def receive(self, input):
        """
        Add logging of state transitions to the wrapped state machine.

        @see: L{IFiniteStateMachine.receive}
        """
        if IRichInput.providedBy(input):
            richInput = unicode(input)
            symbolInput = unicode(input.symbol())
        else:
            richInput = None
            symbolInput = unicode(input)

        action = LOG_FSM_TRANSITION(
            self.logger,
            fsm_identifier=self.identifier,
            fsm_state=unicode(self.state),
            fsm_rich_input=richInput,
            fsm_input=symbolInput)

        with action as theAction:
            output = super(FiniteStateLogger, self).receive(input)
            theAction.addSuccessFields(
                fsm_next_state=unicode(self.state), fsm_output=[unicode(o) for o in output])

        if self._action is not None and self._isTerminal(self.state):
            self._action.addSuccessFields(
                fsm_terminal_state=unicode(self.state))
            self._action.finish()
            self._action = None

        return output