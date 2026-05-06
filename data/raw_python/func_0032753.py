def job_stats_enhanced(job_id):
    """
    Get full job and step stats for job_id
    """
    stats_dict = {}
    with os.popen('bjobs -o "jobid run_time cpu_used  queue slots  stat exit_code start_time estimated_start_time finish_time delimiter=\'|\'" -noheader ' + str(job_id)) as f:
        try:
            line = f.readline()
            cols = line.split('|')
            stats_dict['job_id'] = cols[0]
            if cols[1] != '-':
                stats_dict['wallclock'] = timedelta(
                    seconds=float(cols[1].split(' ')[0]))
            if cols[2] != '-':
                stats_dict['cpu'] = timedelta(
                    seconds=float(cols[2].split(' ')[0]))
            stats_dict['queue'] = cols[3]
            stats_dict['status'] = cols[5]
            stats_dict['exit_code'] = cols[6]
            stats_dict['start'] = cols[7]
            stats_dict['start_time'] = cols[8]
            if stats_dict['status'] in ['DONE', 'EXIT']:
                stats_dict['end'] = cols[9]

            steps = []
            stats_dict['steps'] = steps
        except:
            with os.popen('bhist -l ' + str(job_id)) as f:
                try:
                    output = f.readlines()
                    for line in output:
                        if "Done successfully" in line:
                            stats_dict['status'] = 'DONE'
                            return stats_dict
                        elif "Completed <exit>" in line:
                            stats_dict['status'] = 'EXIT'
                            return stats_dict
                        else:
                            stats_dict['status'] = 'UNKNOWN'
                except Exception as e:
                    print(e)
                    print('LSF: Error reading job stats')
                    stats_dict['status'] = 'UNKNOWN'
    return stats_dict