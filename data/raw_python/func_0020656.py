def exec_with_env(ssh, command, msg='', env={}, **kwargs):
  """

  :param ssh:
  :param command:
  :param msg:
  :param env:
  :param synch:
  :return:
  """
  bash_profile_command = "source .bash_profile > /dev/null 2> /dev/null;"
  env_command = build_os_environment_string(env)
  new_command = bash_profile_command + env_command + command
  if kwargs.get('sync', True):
    return better_exec_command(ssh, new_command, msg)
  else:
    return ssh.exec_command(new_command)