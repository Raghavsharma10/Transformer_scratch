def job_stats_enhanced(job_id):
    """
    Get full job and step stats for job_id
    """
    stats_dict = {}
    with os.popen('qstat -xf ' + str(job_id)) as f:
        try:
            line = f.readline().strip()
            while line:
                if line.startswith('Job Id:'):
                    stats_dict['job_id'] = line.split(':')[1].split('.')[0].strip()
                elif line.startswith('resources_used.walltime'):
                    stats_dict['wallclock'] = get_timedelta(line.split('=')[1])
                elif line.startswith('resources_used.cput'):
                    stats_dict['cpu'] = get_timedelta(line.split('=')[1])
                elif line.startswith('queue'):
                    stats_dict['queue'] = line.split('=')[1].strip()
                elif line.startswith('job_state'):
                    stats_dict['status'] = line.split('=')[1].strip()
                elif line.startswith('Exit_status'):
                    stats_dict['exit_code'] = line.split('=')[1].strip()
                elif line.startswith('Exit_status'):
                    stats_dict['exit_code'] = line.split('=')[1].strip()
                elif line.startswith('stime'):
                    stats_dict['start'] = line.split('=')[1].strip()
                line = f.readline().strip()
            if stats_dict['status'] == 'F' and 'exit_code' not in stats_dict:
                stats_dict['status'] = 'CA'
            elif stats_dict['status'] == 'F' and stats_dict['exit_code'] == '0':
                stats_dict['status'] = 'PBS_F'
        except Exception as e:
            with os.popen('qstat -xaw ' + str(job_id)) as f:
                try:
                    output = f.readlines()
                    for line in output:
                        if str(job_id) in line:
                            cols = line.split()
                            stats_dict['job_id'] = cols[0].split('.')[0]
                            stats_dict['queue'] = cols[2]
                            stats_dict['status'] = cols[9]
                            stats_dict['wallclock'] = get_timedelta(cols[10])
                            return stats_dict
                except Exception as e:
                    print(e)
                    print('PBS: Error reading job stats')
                    stats_dict['status'] = 'UNKNOWN'
    return stats_dict