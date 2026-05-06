def create_event_regressors(self, event_times_indices, covariates = None, durations = None):
        """create_event_regressors creates the part of the design matrix corresponding to one event type. 

            :param event_times_indices: indices in the resampled data, on which the events occurred.
            :type event_times_indices: numpy array, (nr_events)
            :param covariates: covariates belonging to this event type. If None, covariates with a value of 1 for all events are created and used internally.
            :type covariates: numpy array, (nr_events)
            :param durations: durations belonging to this event type. If None, durations with a value of 1 sample for all events are created and used internally.
            :type durations: numpy array, (nr_events)
            :returns: This event type's part of the design matrix.
        """

        # check covariates
        if covariates is None:
            covariates = np.ones(self.event_times_indices.shape)

        # check/create durations, convert from seconds to samples time, and compute mean duration for this event type.
        if durations is None:
            durations = np.ones(self.event_times_indices.shape)
        else:
            durations = np.round(durations*self.deconvolution_frequency).astype(int)
        mean_duration = np.mean(durations)

        # set up output array
        regressors_for_event = np.zeros((self.deconvolution_interval_size, self.resampled_signal_size))

        # fill up output array by looping over events.
        for cov, eti, dur in zip(covariates, event_times_indices, durations):
            valid = True
            if eti < 0:
                self.logger.debug('deconv samples are starting before the data starts.')
                valid = False
            if eti+self.deconvolution_interval_size > self.resampled_signal_size:
                self.logger.debug('deconv samples are continuing after the data stops.')
                valid = False
            if eti > self.resampled_signal_size:
                self.logger.debug('event falls outside of the scope of the data.')
                valid = False

            if valid: # only incorporate sensible events.
                # calculate the design matrix that belongs to this event.
                this_event_design_matrix = (np.diag(np.ones(self.deconvolution_interval_size)) * cov)
                over_durations_dm = np.copy(this_event_design_matrix)
                if dur > 1: # if this event has a non-unity duration, duplicate the stick regressors in the time direction
                    for d in np.arange(1,dur):
                        over_durations_dm[d:] += this_event_design_matrix[:-d]
                    # and correct for differences in durations between different regressor types.
                    over_durations_dm /= mean_duration
                # add the designmatrix for this event to the full design matrix for this type of event.
                regressors_for_event[:,eti:int(eti+self.deconvolution_interval_size)] += over_durations_dm
        
        return regressors_for_event