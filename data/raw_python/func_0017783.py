def paster(opts):
    """Run a paster command from the current directory

Usage:
  datacats paster [-d] [-s NAME] [COMMAND...]

Options:
  -s --site=NAME   Specify a site to run this paster command on [default: primary]
  -d --detach       Run the resulting container in the background

You must be inside a datacats environment to run this. The paster command will
run within your current directory inside the environment. You don't need to
specify the --plugin option. The --config option also need not be specified.
"""
    environment = Environment.load('.')
    environment.require_data()
    environment.start_supporting_containers()

    if not opts['COMMAND']:
        opts['COMMAND'] = ['--', 'help']

    assert opts['COMMAND'][0] == '--'
    return environment.interactive_shell(
        opts['COMMAND'][1:],
        paster=True,
        detach=opts['--detach']
        )