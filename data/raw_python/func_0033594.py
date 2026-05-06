def create_job(cpu_width, time_height):
    """
    :param cpu_width: number of cpus
    :param time_height: amount of time
    :return: the instantiated JobBlock object
    """

    shell_command = stress_string.format(cpu_width, time_height)
    job = JobBlock(cpu_width, time_height)
    job.set_job(subprocess.call, shell_command, shell=True)
    return job