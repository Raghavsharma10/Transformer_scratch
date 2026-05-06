def yn_prompt(text):
    '''
        Takes the text prompt, and presents it, takes only "y" or "n" for
        answers, and returns True or False. Repeats itself on bad input.
    '''

    text = "\n"+ text + "\n('y' or 'n'): "


    while True:
        answer = input(text).strip()
        if answer != 'y' and answer != 'n':
            continue
        elif answer == 'y':
            return True
        elif answer == 'n':
            return False