def get_transaction_history(self, max_wait=5.0):
        """ Download full account history.

            Uses request_transaction_history to get the transaction
            history URL, then polls the given URL until it's ready (or
            the max_wait time is reached) and provides the decoded
            response.

            Parameters
            ----------
            max_wait : float
                The total maximum time to spend waiting for the file to
                be ready; if this is exceeded a failed response will be
                returned.  This is not guaranteed to be strictly
                followed, as one last attempt will be made to check the
                file before giving up.

            See more:
            http://developer.oanda.com/rest-live/transaction-history/#getFullAccountHistory
            http://developer.oanda.com/rest-live/transaction-history/#transactionTypes
        """
        url = self.request_transaction_history()
        if not url:
            return False

        ready = False
        start = time()
        delay = 0.1
        while not ready and delay:
            response = requests.head(url)
            ready = response.ok
            if not ready:
                sleep(delay)
                time_remaining = max_wait - time() + start
                max_delay = max(0., time_remaining - .1)
                delay = min(delay * 2, max_delay)

        if not ready:
            return False

        response = requests.get(url)
        try:
            with ZipFile(BytesIO(response.content)) as container:
                files = container.namelist()
                if not files:
                    log.error('Transaction ZIP has no files.')
                    return False
                history = container.open(files[0])
                raw = history.read().decode('ascii')
        except BadZipfile:
            log.error('Response is not a valid ZIP file', exc_info=True)
            return False

        return json.loads(raw, **self.json_options)