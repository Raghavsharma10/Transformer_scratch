def remote_server_command(command, environment, user_profile, **kwargs):
    """
      Wraps web_command function with docker bindings needed to connect to
      a remote server (such as datacats.com) and run commands there
      (for example, when you want to copy your catalog to that server).

      The files binded to the docker image include the user's ssh credentials:
          ssh_config file,
          rsa and rsa.pub user keys
          known_hosts whith public keys of the remote server (if known)

      The **kwargs (keyword arguments) are passed on to the web_command call
      intact, see the web_command's doc string for details
    """

    if environment.remote_server_key:
        temp = tempfile.NamedTemporaryFile(mode="wb")
        temp.write(environment.remote_server_key)
        temp.seek(0)
        known_hosts = temp.name
    else:
        known_hosts = get_script_path('known_hosts')

    binds = {
        user_profile.profiledir + '/id_rsa': '/root/.ssh/id_rsa',
        known_hosts: '/root/.ssh/known_hosts',
        get_script_path('ssh_config'): '/etc/ssh/ssh_config'
    }

    if kwargs.get("include_project_dir", None):
        binds[environment.target] = '/project'
        del kwargs["include_project_dir"]

    kwargs["ro"] = binds
    try:
        web_command(command, **kwargs)
    except WebCommandError as e:
        e.user_description = 'Sending a command to remote server failed'
        raise e