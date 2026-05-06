def write_summary(all_procs, summary_file):
    """
    Write a summary of all run processes to summary_file in tab-delimited
    format.
    """
    if not summary_file:
        return

    with summary_file:
        writer = csv.writer(summary_file, delimiter='\t', lineterminator='\n')
        writer.writerow(('directory', 'command', 'start_time', 'end_time',
            'run_time', 'exit_status', 'result'))
        rows = ((p.working_dir, ' '.join(p.command), p.start_time, p.end_time,
                p.running_time, p.return_code, p.status) for p in all_procs)
        writer.writerows(rows)