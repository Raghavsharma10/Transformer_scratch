def calculate_start_time(df):
    """Calculate the star_time per read.

    Time data is either
    a "time" (in seconds, derived from summary files) or
    a "timestamp" (in UTC, derived from fastq_rich format)
    and has to be converted appropriately in a datetime format time_arr

    For both the time_zero is the minimal value of the time_arr,
    which is then used to subtract from all other times

    In the case of method=track (and dataset is a column in the df) then this
    subtraction is done per dataset
    """
    if "time" in df:
        df["time_arr"] = pd.Series(df["time"], dtype='datetime64[s]')
    elif "timestamp" in df:
        df["time_arr"] = pd.Series(df["timestamp"], dtype="datetime64[ns]")
    else:
        return df
    if "dataset" in df:
        for dset in df["dataset"].unique():
            time_zero = df.loc[df["dataset"] == dset, "time_arr"].min()
            df.loc[df["dataset"] == dset, "start_time"] = \
                df.loc[df["dataset"] == dset, "time_arr"] - time_zero
    else:
        df["start_time"] = df["time_arr"] - df["time_arr"].min()
    return df.drop(["time", "timestamp", "time_arr"], axis=1, errors="ignore")