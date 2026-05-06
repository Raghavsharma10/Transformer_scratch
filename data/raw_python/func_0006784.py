def disable_requiretty_on_sudoers(log=False):
    """ allow sudo calls through ssh without a tty """
    if log:
        bookshelf2.logging_helpers.log_green(
            'disabling requiretty on sudo calls')

    comment_line('/etc/sudoers',
                 '^Defaults.*requiretty', use_sudo=True)
    return True