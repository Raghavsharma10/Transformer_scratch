def check_valid_time_and_sort(df, timescol, days=5, warning=True):
    """Check if the data contains reads created within the same `days` timeframe.

    if not, print warning and only return part of the data which is within `days` days
    Resetting the index twice to get also an "index" column for plotting the cum_yield_reads plot
    """
    timediff = (df[timescol].max() - df[timescol].min()).days
    if timediff < days:
        return df.sort_values(timescol).reset_index(drop=True).reset_index()
    else:
        if warning:
            sys.stderr.write(
                "\nWarning: data generated is from more than {} days.\n".format(str(days)))
            sys.stderr.write("Likely this indicates you are combining multiple runs.\n")
            sys.stderr.write(
                "Plots based on time are invalid and therefore truncated to first {} days.\n\n"
                .format(str(days)))
            logging.warning("Time plots truncated to first {} days: invalid timespan: {} days"
                            .format(str(days), str(timediff)))
        return df[df[timescol] < timedelta(days=days)] \
            .sort_values(timescol) \
            .reset_index(drop=True) \
            .reset_index()