def remove_file(master_ip, filename, spark_on_toil):
    """
    Remove the given file from hdfs with master at the given IP address

    :type masterIP: MasterAddress
    """
    master_ip = master_ip.actual

    ssh_call = ['ssh', '-o', 'StrictHostKeyChecking=no', master_ip]

    if spark_on_toil:
        output = check_output(ssh_call + ['docker', 'ps'])
        container_id = next(line.split()[0] for line in output.splitlines() if 'apache-hadoop-master' in line)
        ssh_call += ['docker', 'exec', container_id]

    try:
        check_call(ssh_call + ['hdfs', 'dfs', '-rm', '-r', '/' + filename])
    except:
        pass