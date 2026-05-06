def __stage_order(self):
        """
        Add all the loans to the LC order session
        """

        # Skip staging...probably not a good idea...you've been warned
        if self.__already_staged is True and self.__i_know_what_im_doing is True:
            self.__log('Not staging the order...I hope you know what you\'re doing...'.format(len(self.loans)))
            return

        self.__log('Staging order for {0} loan notes...'.format(len(self.loans)))

        # Create a fresh order session
        self.lc.session.clear_session_order()

        #
        # Stage all the loans to the order
        #
        loan_ids = self.loans.keys()
        self.__log('Staging loans {0}'.format(loan_ids))

        # LendingClub requires you to search for the loans before you can stage them
        f = FilterByLoanID(loan_ids)
        results = self.lc.search(f, limit=len(self.loans))
        if len(results['loans']) == 0 or results['totalRecords'] != len(self.loans):
            raise LendingClubError('Could not stage the loans. The number of loans in your batch does not match totalRecords. {0} != {1}'.format(len(self.loans), results['totalRecords']), results)

        # Stage each loan
        for loan_id, amount in self.loans.iteritems():
            payload = {
                'method': 'addToPortfolio',
                'loan_id': loan_id,
                'loan_amount': amount,
                'remove': 'false'
            }
            response = self.lc.session.get('/data/portfolio', query=payload)
            json_response = response.json()

            # Ensure it was successful before moving on
            if not self.lc.session.json_success(json_response):
                raise LendingClubError('Could not stage loan {0} on the order: {1}'.format(loan_id, response.text), response)

        #
        # Add all staged loans to the order
        #
        payload = {
            'method': 'addToPortfolioNew'
        }
        response = self.lc.session.get('/data/portfolio', query=payload)
        json_response = response.json()

        if self.lc.session.json_success(json_response):
            self.__log(json_response['message'])
            return True
        else:
            raise self.__log('Could not add loans to the order: {0}'.format(response.text))
            raise LendingClubError('Could not add loans to the order', response.text)