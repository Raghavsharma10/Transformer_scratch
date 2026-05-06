def print_report(runner_results):
    """
    Print collated report with output and errors if any
    """
    error_report = collections.defaultdict(list)
    output_report = collections.defaultdict(list)
    success_report = list()
    
    for runner_info in runner_results:
        hostname = runner_info['console']
        error = runner_info['error']
        output = runner_info['output']
        
        if error:
            error_report[error].append(hostname)
        elif output:
            output_report[output].append(hostname)
        else:
            success_report.append(hostname)
            
    if error_report:
        print("Errors : ")
        for error in error_report:
            print("{0} -- [{1}] {2}".format(error.strip(), len(error_report[error]), ", ".join(error_report[error])))
            print()
    
    if output_report:        
        for output in output_report:
            print("{0} -- [{1}] {2}".format(output, len(output_report[output]), ", ".join(output_report[output])))
    
    if success_report:
        print("Completed config on {0} hosts".format(len(success_report)))