def job_stats_enhanced(job_id):
    """
    Get full job and step stats for job_id
    """
    stats_dict = {}
    with os.popen('sacct --noheader --format JobId,Elapsed,TotalCPU,Partition,NTasks,AveRSS,State,ExitCode,start,end -P -j ' + str(job_id)) as f:
        try:
            line = f.readline()
            if line in ["SLURM accounting storage is disabled",
                        "slurm_load_job error: Invalid job id specified"]:
                raise
            cols = line.split('|')
            stats_dict['job_id'] = cols[0]
            stats_dict['wallclock'] = get_timedelta(cols[1])
            stats_dict['cpu'] = get_timedelta(cols[2])
            stats_dict['queue'] = cols[3]
            stats_dict['status'] = cols[6]
            stats_dict['exit_code'] = cols[7].split(':')[0]
            stats_dict['start'] = cols[8]
            stats_dict['end'] = cols[9]

            steps = []
            for line in f:
                step = {}
                cols = line.split('|')
                step_val = cols[0].split('.')[1]
                step['step'] = step_val
                step['wallclock'] = get_timedelta(cols[1])
                step['cpu'] = get_timedelta(cols[2])
                step['ntasks'] = cols[4]
                step['status'] = cols[6]
                step['exit_code'] = cols[7].split(':')[0]
                step['start'] = cols[8]
                step['end'] = cols[9]
                steps.append(step)
            stats_dict['steps'] = steps
        except:
            with os.popen('squeue -h -j %s' % str(job_id)) as f:
                try:
                    for line in f:
                        new_line = re.sub(' +', ' ', line.strip())
                        job_id = int(new_line.split(' ')[0])
                        state = new_line.split(' ')[4]
                        stats_dict['job_id'] = str(job_id)
                        stats_dict['status'] = state
                except:
                    print('SLURM: Error reading job stats')
                    stats_dict['status'] = 'UNKNOWN'
    with os.popen('squeue --format %%S -h -j ' + str(job_id)) as f:
        try:
            line = f.readline()
            if len(line) > 0:
                stats_dict['start_time'] = line
            else:
                stats_dict['start_time'] = ""
        except:
            print('SLURM: Error getting start time')
    return stats_dict