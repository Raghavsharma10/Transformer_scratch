def _print_result(case, summary):
    """ Show some statistics from the run """
    for case, case_data in summary.items():
        for dof, data in case_data.items():
            print("    " + case + " " + dof)
            print("    -------------------")
            for header, val in data.items():
                print("    " + header + " : " + str(val))
            print("")