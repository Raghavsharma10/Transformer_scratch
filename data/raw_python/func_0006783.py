def disable_env_reset_on_sudo(log=False):
    """ updates /etc/sudoers so that users from %wheel keep their
        environment when executing a sudo call
    """
    if log:
        bookshelf2.logging_helpers.log_green('disabling env reset on sudo')

    file_append('/etc/sudoers',
                'Defaults:%wheel !env_reset,!secure_path',
                use_sudo=True,
                partial=True)
    return True