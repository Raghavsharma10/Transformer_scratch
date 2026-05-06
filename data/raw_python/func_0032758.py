def gtpswd(prompt, confirmPassword):
    """
    Temporary wrapper for Twisted's getPassword until a version that supports
    customizing the 'confirm' prompt is released.
    """
    try:
        return util.getPassword(prompt=prompt,
                                confirmPrompt=confirmPassword,
                                confirm=True)
    except TypeError:
        return util.getPassword(prompt=prompt,
                                confirm=True)