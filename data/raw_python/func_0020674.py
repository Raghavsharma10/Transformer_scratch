def install(self, unique_id, configs=None):
    """
    Copies the executable to the remote machine under install path. Inspects the configs for the possible keys
    'hostname': the host to install on
    'install_path': the location on the remote host
    'executable': the executable to copy
    'no_copy': if this config is passed in and true then this method will not copy the executable assuming that it is
    already installed
    'post_install_cmds': an optional list of commands that should be executed on the remote machine after the
     executable has been installed. If no_copy is set to true, then the post install commands will not be run.

    If the unique_id is already installed on a different host, this will perform the cleanup action first.
    If either 'install_path' or 'executable' are provided the new value will become the default.

    :param unique_id:
    :param configs:
    :return:
    """

    # the following is necessay to set the configs for this function as the combination of the
    # default configurations and the parameter with the parameter superceding the defaults but
    # not modifying the defaults
    if configs is None:
      configs = {}
    tmp = self.default_configs.copy()
    tmp.update(configs)
    configs = tmp

    hostname = None
    is_tarfile = False
    is_zipfile = False
    if unique_id in self.processes and  'hostname' in configs:
        self.uninstall(unique_id, configs)
        hostname = configs['hostname']
    elif 'hostname' in configs:
      hostname = configs['hostname']
    elif unique_id not in self.processes:
      # we have not installed this unique_id before and no hostname is provided in the configs so raise an error
      raise DeploymentError("hostname was not provided for unique_id: " + unique_id)

    env = configs.get("env", {})
    install_path = configs.get('install_path') or self.default_configs.get('install_path')
    pid_file = configs.get('pid_file') or self.default_configs.get('pid_file')
    if install_path is None:
      logger.error("install_path was not provided for unique_id: " + unique_id)
      raise DeploymentError("install_path was not provided for unique_id: " + unique_id)
    if not configs.get('no_copy', False):
      with get_ssh_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ssh:
        log_output(better_exec_command(ssh, "mkdir -p {0}".format(install_path),
                                       "Failed to create path {0}".format(install_path)))
        log_output(better_exec_command(ssh, "chmod 755 {0}".format(install_path),
                                       "Failed to make path {0} writeable".format(install_path)))
        executable = configs.get('executable') or self.default_configs.get('executable')
        if executable is None:
          logger.error("executable was not provided for unique_id: " + unique_id)
          raise DeploymentError("executable was not provided for unique_id: " + unique_id)

        #if the executable is in remote location copy to local machine
        copy_from_remote_location = False;

        if (":" in executable):
          copy_from_remote_location = True

          if ("http" not in executable):
            remote_location_server = executable.split(":")[0]
            remote_file_path = executable.split(":")[1] 
            remote_file_name = os.path.basename(remote_file_path)

            local_temp_file_name = os.path.join(configs.get("tmp_dir","/tmp"),remote_file_name)
          
            if not os.path.exists(local_temp_file_name):
              with get_sftp_client(remote_location_server,username=runtime.get_username(), password=runtime.get_password()) as ftp:
                try:
                  ftp.get(remote_file_path, local_temp_file_name)
                  executable = local_temp_file_name
                except:
                  raise DeploymentError("Unable to load file from remote server " + executable)
          #use urllib for http copy
          else:    
              remote_file_name = executable.split("/")[-1]
              local_temp_file_name = os.path.join(configs.get("tmp_dir","/tmp"),remote_file_name)
              if not os.path.exists(local_temp_file_name):
                try:
                  urllib.urlretrieve (executable, local_temp_file_name)
                except:
                  raise DeploymentError("Unable to load file from remote server " + executable)
              executable = local_temp_file_name    

        try:                     
          exec_name = os.path.basename(executable)
          install_location = os.path.join(install_path, exec_name)
          with get_sftp_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ftp:
            ftp.put(executable, install_location)
        except:
            raise DeploymentError("Unable to copy executable to install_location:" + install_location)
        finally:
          #Track if its a tarfile or zipfile before deleting it in case the copy to remote location fails
          is_tarfile = tarfile.is_tarfile(executable)
          is_zipfile = zipfile.is_zipfile(executable)
          if (copy_from_remote_location and not configs.get('cache',False)):
            os.remove(executable)       

        # only supports tar and zip (because those modules are provided by Python's standard library)
        if configs.get('extract', False) or self.default_configs.get('extract', False):
          if is_tarfile:
            log_output(better_exec_command(ssh, "tar -xf {0} -C {1}".format(install_location, install_path),
                                           "Failed to extract tarfile {0}".format(exec_name)))
          elif is_zipfile:
            log_output(better_exec_command(ssh, "unzip -o {0} -d {1}".format(install_location, install_path),
                                           "Failed to extract zipfile {0}".format(exec_name)))
          else:
            logger.error(executable + " is not a supported filetype for extracting")
            raise DeploymentError(executable + " is not a supported filetype for extracting")
        post_install_cmds = configs.get('post_install_cmds', False) or self.default_configs.get('post_install_cmds', [])
        for cmd in post_install_cmds:
          relative_cmd = "cd {0}; {1}".format(install_path, cmd)
          log_output(exec_with_env(ssh, relative_cmd,
                                         msg="Failed to execute post install command: {0}".format(relative_cmd), env=env))
    self.processes[unique_id] = Process(unique_id, self.service_name, hostname, install_path)
    self.processes[unique_id].pid_file = pid_file