def salt_master(project, target, module, args=None, kwargs=None):
    """
    Execute a `salt` command in the head node
    """
    client = project.cluster.head.ssh_client

    cmd = ['salt']
    cmd.extend(generate_salt_cmd(target, module, args, kwargs))
    cmd.append('--timeout=300')
    cmd.append('--state-output=mixed')
    cmd = ' '.join(cmd)

    output = client.exec_command(cmd, sudo=True)
    if output['exit_code'] == 0:
        return output['stdout']
    else:
        return output['stderr']