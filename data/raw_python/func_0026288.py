def wrap_prompts_class(Klass):
    """
    Wrap an IPython's Prompt class

    This is needed in order for Prompt to inject the correct escape sequences
    at the right positions for shell integrations.

    """

    try:
        from prompt_toolkit.token import ZeroWidthEscape
    except ImportError:
        return Klass

    class ITerm2IPythonPrompt(Klass):

        def in_prompt_tokens(self, cli=None):
            return  [
                     (ZeroWidthEscape, last_status(self.shell)+BEFORE_PROMPT),
                    ]+\
                    super(ITerm2IPythonPrompt, self).in_prompt_tokens(cli)+\
                    [(ZeroWidthEscape, AFTER_PROMPT)]


    return ITerm2IPythonPrompt