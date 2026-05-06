def shell(environment, opts):
    """Run a command or interactive shell within this environment

Usage:
  datacats [-d] [-s NAME] shell [ENVIRONMENT [COMMAND...]]

Options:
  -d --detach       Run the resulting container in the background
  -s --site=NAME   Specify a site to run the shell on [default: primary]

ENVIRONMENT may be an environment name or a path to an environment directory.
Default: '.'
"""
    environment.require_data()
    environment.start_supporting_containers()
    return environment.interactive_shell(
        opts['COMMAND'],
        detach=opts['--detach']
    )