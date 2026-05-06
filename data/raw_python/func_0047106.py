def repstring_penalty(reps, intensities, desired_reps, desired_intensity,
                          minimum_rep):
        """Penalty function which calculates how "bad" a set of
        reps and intensities is, compared to the desired repetitions,
        the desired intensity level and the minimum repetitions.
        Advanced users may substitute this function for their own version.
    
        Parameters
        ----------
        reps
            A list of repetitions (sorted), e.g. [8, 6, 5, 2].
        intensities
            A list of intensities corresponding to the repetitions,
            e.g. [64.7, 72.3, 76.25, 88.7].
        desired_reps
            Desired number of repetitions in total, e.g. 25.
        desired_intensity
            The desired average intensity, e.g. 75.
        minimum_rep
            The minimum repetition which is allowed, e.g. 2.
    
    
        Returns
        -------
        float
            A penalty, a positive real number.
    
    
        Examples
        -------
        >>> desired_reps = 25
        >>> desired_intensity = 75
        >>> minimum_rep = 1
        >>> high = Program().repstring_penalty([8, 8, 8], [60, 60, 60], 
        ...                              desired_reps, desired_intensity, 
        ...                              minimum_rep)
        >>> low = Program().repstring_penalty([8, 6, 5, 4, 2], [64, 72, 75, 80, 88], 
        ...                              desired_reps, desired_intensity, 
        ...                              minimum_rep)
        >>> high > low
        True
        """
        # Punish when the mean intensity is far from the desired one
        desired = desired_intensity
        error1 = abs(statistics.mean(intensities) - desired)

        # Punish when the repetitions are far from the desired amount
        error2 = abs(sum(reps) - desired_reps)

        # Punish when the spread of repetitions is large
        error3 = spread(reps)

        # Punish deviation from the minimum reptition
        error4 = abs(min(reps) - minimum_rep)

        # Take a linear combination and return
        return sum([2 * error1, 0.5 * error2, 2.5 * error3, 0.5 * error4])