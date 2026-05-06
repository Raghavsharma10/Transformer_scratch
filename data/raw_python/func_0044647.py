def load_config(config_file, default_config_file, **kwargs):
    '''load the configuration

    Configuration variables are delivered to applications exclusively through
    the environment. They get into the environment either from a specified
    configuration file, from a default configuration file, from an environment
    variable, or from a kwarg specified to this function.

    Configuration variables are applied with the following precedence (lowest
    to highest):

    - config file: the format of the config file is a typical environment file
      with individual lines like DATALAKE_FOO=bar. By convention, all variables
      start with either DATALAKE_ or AWS_.

    - environment variables: The variable names are the same as what would be
      written in a config file.

    - kwargs: additional configuration variables to apply, subject to some
      conventions. Specifically, kwargs are lowercase. A kwarg called `foo`
      maps to a configuration variable called `DATALAKE_FOO`. The only
      exception to this is a kwarg that starts with `aws_`. That is, a kwarg
      called `aws_baz` would map to a configuration variable called `AWS_BAZ`.


    Args:

    - config_file: the configuration file to load. If it is None,
      default_config_file will be examined. If it is not None and does not
      exist an InsufficientConfiguration exception is thrown.

    - default_config_file: the file to try if config_file is None. If
      default_config_file is None or does not exist, it is simply ignored. No
      exceptions are thrown.

    - kwargs: key=value pairs.

    '''
    if config_file and not os.path.exists(config_file):
        msg = 'config file {} does not exist'.format(config_file)
        raise InsufficientConfiguration(msg)

    if config_file is None and \
       default_config_file is not None and \
       os.path.exists(default_config_file):
        config_file = default_config_file

    if config_file is not None:
        load_dotenv(config_file)

    _update_environment(**kwargs)