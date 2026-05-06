def salt_ssh(project, target, module, args=None, kwargs=None):
    """
    Execute a `salt-ssh` command
    """
    cmd = ['salt-ssh']
    cmd.extend(generate_salt_cmd(target, module, args, kwargs))
    cmd.append('--state-output=mixed')
    cmd.append('--roster-file=%s' % project.roster_path)
    cmd.append('--config-dir=%s' % project.salt_ssh_config_dir)
    cmd.append('--ignore-host-keys')
    cmd.append('--force-color')
    cmd = ' '.join(cmd)
    logger.debug('salt-ssh cmd: %s', cmd)

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0 or err:
        raise Exception(err)
    return out + err