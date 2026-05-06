def parse_log(file_path):
    """
    Parse a CISM output log and extract some information.

    Args:
        file_path: absolute path to the log file

    Return:
        A dictionary created by the elements object corresponding to
        the results of the bit for bit testing
    """
    if not os.path.isfile(file_path):
        return elements.error("Output Log", "Could not open file: " + file_path.split(os.sep)[-1])

    headers = ["Converged Iterations",
               "Avg. Iterations to Converge",
               "Processor Count",
               "Dycore Type"]

    with open(file_path, 'r') as f:
        dycore_types = {"0": "Glide",
                        "1": "Glam",
                        "2": "Glissade",
                        "3": "Albany_felix",
                        "4": "BISICLES"}
        curr_step = 0
        proc_count = 0
        iter_number = 0
        converged_iters = []
        iters_to_converge = []
        for line in f:
            split = line.split()
            if ('CISM dycore type' in line):
                if line.split()[-1] == '=':
                    dycore_type = dycore_types[next(f).strip()]
                else:
                    dycore_type = dycore_types[line.split()[-1]]
            elif ('total procs' in line):
                proc_count += int(line.split()[-1])
            elif ('Nonlinear Solver Step' in line):
                curr_step = int(line.split()[4])
            elif ('Compute ice velocities, time = ' in line):
                converged_iters.append(curr_step)
                curr_step = float(line.split()[-1])
            elif ('"SOLVE_STATUS_CONVERGED"' in line):
                split = line.split()
                iters_to_converge.append(int(split[split.index('"SOLVE_STATUS_CONVERGED"') + 2]))
            elif ("Compute dH/dt" in line):
                iters_to_converge.append(int(iter_number))
            elif len(split) > 0 and split[0].isdigit():
                iter_number = split[0]
        if iters_to_converge == []:
            iters_to_converge.append(int(iter_number))
    data = {
        "Dycore Type": dycore_type,
        "Processor Count": proc_count,
        "Converged Iterations": len(converged_iters),
        "Avg. Iterations to Converge": np.mean(iters_to_converge)
    }
    return elements.table("Output Log", headers, data)