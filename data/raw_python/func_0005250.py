def autorun():
    '''
    Call the run method of the decorated class if the current file is the main file
    '''
    def wrapper(cls):

        import inspect
        if inspect.getmodule(cls).__name__ == "__main__":
            cls().run()
        return cls

    return wrapper