def ask(question, no_input=False):
    """Display a Y/n question prompt, and return a boolean"""
    if no_input:
        return True
    else:
        input_ = input('%s [Y/n] ' % question)
        input_ = input_.strip().lower()
        if input_ in ('y', 'yes', ''):
            return True
        if input_ in ('n', 'no'):
            return False
        print('Invalid selection')