def search_my_notes(self, loan_id=None, order_id=None, grade=None, portfolio_name=None, status=None, term=None):
        """
        Search for notes you are invested in. Use the parameters to define how to search.
        Passing no parameters is the same as calling `my_notes(get_all=True)`

        Parameters
        ----------
        loan_id : int, optional
            Search for notes for a specific loan. Since a loan is broken up into a pool of notes, it's possible
            to invest multiple notes in a single loan
        order_id : int, optional
            Search for notes from a particular investment order.
        grade : {A, B, C, D, E, F, G}, optional
            Match by a particular loan grade
        portfolio_name : string, optional
            Search for notes in a portfolio with this name (case sensitive)
        status : string, {issued, in-review, in-funding, current, charged-off, late, in-grace-period, fully-paid}, optional
            The funding status string.
        term : {60, 36}, optional
            Term length, either 60 or 36 (for 5 year and 3 year, respectively)

        Returns
        -------
        dict
            A dictionary with a list of matching notes on the `loans` key
        """
        assert grade is None or type(grade) is str, 'grade must be a string'
        assert portfolio_name is None or type(portfolio_name) is str, 'portfolio_name must be a string'

        index = 0
        found = []
        sort_by = 'orderId' if order_id is not None else 'loanId'
        group_id = order_id if order_id is not None else loan_id   # first match by order, then by loan

        # Normalize grade
        if grade is not None:
            grade = grade[0].upper()

        # Normalize status
        if status is not None:
            status = re.sub('[^a-zA-Z\-]', ' ', status.lower())  # remove all non alpha characters
            status = re.sub('days', ' ', status)  # remove days
            status = re.sub('\s+', '-', status.strip())  # replace spaces with dash
            status = re.sub('(^-+)|(-+$)', '', status)

        while True:
            notes = self.my_notes(start_index=index, sort_by=sort_by)

            if notes['result'] != 'success':
                break

            # If the first note has a higher ID, we've passed it
            if group_id is not None and notes['loans'][0][sort_by] > group_id:
                break

            # If the last note has a higher ID, it could be in this record set
            if group_id is None or notes['loans'][-1][sort_by] >= group_id:
                for note in notes['loans']:

                    # Order ID, no match
                    if order_id is not None and note['orderId'] != order_id:
                        continue

                    # Loan ID, no match
                    if loan_id is not None and note['loanId'] != loan_id:
                        continue

                    # Grade, no match
                    if grade is not None and note['rate'][0] != grade:
                        continue

                    # Portfolio, no match
                    if portfolio_name is not None and note['portfolioName'][0] != portfolio_name:
                        continue

                    # Term, no match
                    if term is not None and note['loanLength'] != term:
                        continue

                    # Status
                    if status is not None:
                        # Normalize status message
                        nstatus = re.sub('[^a-zA-Z\-]', ' ', note['status'].lower())  # remove all non alpha characters
                        nstatus = re.sub('days', ' ', nstatus)  # remove days
                        nstatus = re.sub('\s+', '-', nstatus.strip())  # replace spaces with dash
                        nstatus = re.sub('(^-+)|(-+$)', '', nstatus)

                        # No match
                        if nstatus != status:
                            continue

                    # Must be a match
                    found.append(note)

            index += 100

        return found