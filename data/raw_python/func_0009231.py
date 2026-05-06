def validate_one(self, loan):
        """
        Validate a single loan result record against the filters

        Parameters
        ----------
        loan : dict
            A single loan note record

        Returns
        -------
        boolean
            True or raises FilterValidationError

        Raises
        ------
        FilterValidationError
            If the loan does not match the filter criteria
        """
        assert type(loan) is dict, 'loan parameter must be a dictionary object'

        # Map the loan value keys to the filter keys
        req = {
            'loanGUID': 'loan_id',
            'loanGrade': 'grade',
            'loanLength': 'term',
            'loanUnfundedAmount': 'progress',
            'loanAmountRequested': 'progress',
            'alreadyInvestedIn': 'exclude_existing',
            'purpose': 'loan_purpose',
        }

        # Throw an error if the loan does not contain one of the criteria keys that this filter has
        for key, criteria in req.iteritems():
            if criteria in self and key not in loan:
                raise FilterValidationError('Loan does not have a "{0}" value.'.format(key), loan, criteria)

        # Loan ID
        if 'loan_id' in self:
            loan_ids = str(self['loan_id']).split(',')
            if str(loan['loanGUID']) not in loan_ids:
                raise FilterValidationError('Did not meet filter criteria for loan ID. {0} does not match {1}'.format(loan['loanGUID'], self['loan_id']), loan=loan, criteria='loan ID')

        # Grade
        grade = loan['loanGrade'][0]  # Extract the letter portion of the loan
        if 'grades' in self and self['grades']['All'] is not True:
            if grade not in self['grades']:
                raise FilterValidationError('Loan grade "{0}" is unknown'.format(grade), loan, 'grade')
            elif self['grades'][grade] is False:
                raise FilterValidationError(loan=loan, criteria='grade')

        # Term
        if 'term' in self and self['term'] is not None:
            if loan['loanLength'] == 36 and self['term']['Year3'] is False:
                raise FilterValidationError(loan=loan, criteria='loan term')
            elif loan['loanLength'] == 60 and self['term']['Year5'] is False:
                raise FilterValidationError(loan=loan, criteria='loan term')

        # Progress
        if 'funding_progress' in self:
            loan_progress = (1 - (loan['loanUnfundedAmount'] / loan['loanAmountRequested'])) * 100
            if self['funding_progress'] > loan_progress:
                raise FilterValidationError(loan=loan, criteria='funding progress')

        # Exclude existing
        if 'exclude_existing' in self:
            if self['exclude_existing'] is True and loan['alreadyInvestedIn'] is True:
                raise FilterValidationError(loan=loan, criteria='exclude loans you are invested in')

        # Loan purpose (either an array or single value)
        if 'loan_purpose' in self and loan['purpose'] is not False:
            purpose = self['loan_purpose']
            if type(purpose) is not dict:
                purpose = {purpose: True}

            if 'All' not in purpose or purpose['All'] is False:
                if loan['purpose'] not in purpose:
                    raise FilterValidationError(loan=loan, criteria='loan purpose')

        return True