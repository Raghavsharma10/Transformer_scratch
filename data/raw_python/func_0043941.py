def interactive_merge_conflict_handler(self, exception):
        """
        Give the operator a chance to interactively resolve merge conflicts.

        :param exception: An :exc:`~executor.ExternalCommandFailed` object.
        :returns: :data:`True` if the operator has interactively resolved any
                  merge conflicts (and as such the merge error doesn't need to
                  be propagated), :data:`False` otherwise.

        This method checks whether :data:`sys.stdin` is connected to a terminal
        to decide whether interaction with an operator is possible. If it is
        then an interactive terminal prompt is used to ask the operator to
        resolve the merge conflict(s). If the operator confirms the prompt, the
        merge error is swallowed instead of propagated. When :data:`sys.stdin`
        is not connected to a terminal or the operator denies the prompt the
        merge error is propagated.
        """
        if connected_to_terminal(sys.stdin):
            logger.info(compact("""
                It seems that I'm connected to a terminal so I'll give you a
                chance to interactively fix the merge conflict(s) in order to
                avoid propagating the merge error. Please mark or stage your
                changes but don't commit the result just yet (it will be done
                for you).
            """))
            while True:
                if prompt_for_confirmation("Ignore merge error because you've resolved all conflicts?"):
                    if self.merge_conflicts:
                        logger.warning("I'm still seeing merge conflicts, please double check! (%s)",
                                       concatenate(self.merge_conflicts))
                    else:
                        # The operator resolved all conflicts.
                        return True
                else:
                    # The operator wants us to propagate the error.
                    break
        return False