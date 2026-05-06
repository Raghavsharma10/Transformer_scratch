def _summarize_result(result, summary):
    """ Trim out some data to return for the index page """
    if "Bit for Bit" not in summary:
        summary["Bit for Bit"] = [0, 0]
    if "Configurations" not in summary:
        summary["Configurations"] = [0, 0]
    if "Std. Out Files" not in summary:
        summary["Std. Out Files"] = 0

    # Get the number of bit for bit failures
    total_count = 0
    failure_count = 0
    summary_data = None
    for elem in result:
        if elem["Type"] == "Bit for Bit" and "Data" in elem:
            elem_data = elem["Data"]
            summary_data = summary["Bit for Bit"]
            total_count += 1
            for var in six.iterkeys(elem_data):
                if elem_data[var]["Max Error"] != 0:
                    failure_count += 1
                    break
    if summary_data is not None:
        summary_data = np.add(summary_data, [total_count-failure_count, total_count]).tolist()
        summary["Bit for Bit"] = summary_data

    # Get the number of config matches
    summary_data = None
    total_count = 0
    failure_count = 0
    for elem in result:
        if elem["Title"] == "Configuration Comparison" and elem["Type"] == "Diff":
            elem_data = elem["Data"]
            summary_data = summary["Configurations"]
            total_count += 1
            failed = False
            for section_name, varlist in elem_data.items():
                for var, val in varlist.items():
                    if not val[0]:
                        failed = True
            if failed:
                failure_count += 1
    if summary_data is not None:
        success_count = total_count - failure_count
        summary_data = np.add(summary_data, [success_count, total_count]).tolist()
        summary["Configurations"] = summary_data

    # Get the number of files parsed
    for elem in result:
        if elem["Title"] == "Output Log" and elem["Type"] == "Table":
            summary["Std. Out Files"] += 1
            break
    return summary