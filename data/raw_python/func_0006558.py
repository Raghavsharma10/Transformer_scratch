def regression():
    """
    Run regression testing - lint and then run all tests.
    """
    # HACK: Start using hitchbuildpy to get around this.
    Command("touch", DIR.project.joinpath("pathquery", "__init__.py").abspath()).run()
    storybook = _storybook({}).only_uninherited()
    #storybook.with_params(**{"python version": "2.7.10"})\
             #.ordered_by_name().play()
    Command("touch", DIR.project.joinpath("pathquery", "__init__.py").abspath()).run()
    storybook.with_params(**{"python version": "3.5.0"}).ordered_by_name().play()
    lint()