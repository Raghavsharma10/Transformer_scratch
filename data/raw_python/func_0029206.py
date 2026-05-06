def get_invoke(command):
    """
    Get the Cmd main method from the click command
    :param command: The click Command object
    :return: the do_* method for Cmd
    :rtype: function
    """

    assert isinstance(command, click.Command)

    def invoke_(self, arg):  # pylint: disable=unused-argument
        try:
            command.main(args=shlex.split(arg),
                         prog_name=command.name,
                         standalone_mode=False,
                         parent=self.ctx)
        except click.ClickException as e:
            # Show the error message
            e.show()
        except click.Abort:
            # We got an EOF or Keyboard interrupt.  Just silence it
            pass
        except SystemExit:
            # Catch this an return the code instead. All of click's help commands do a sys.exit(),
            # and that's not ideal when running in a shell.
            pass
        except Exception as e:
            traceback.print_exception(type(e), e, None)
            logger.warning(traceback.format_exc())

        # Always return False so the shell doesn't exit
        return False

    invoke_ = update_wrapper(invoke_, command.callback)
    invoke_.__name__ = 'do_%s' % command.name
    return invoke_