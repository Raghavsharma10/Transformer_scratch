def _parse_transactions(self, response):
        """
        This method parses the CSV output in `get_transactions`
        to generate a usable list of transactions that use native
        python data types
        """
        transactions = list()

        if response:
            f = StringIO(response)
            reader = csv.DictReader(f)

            for line in reader:
                txn = {}
                txn['date'] = datetime.strptime(line['Date'], '%d/%m/%Y %H:%M:%S')
                txn['description'] = line['Description']
                txn['amount'] = float(line['Amount'].replace(',', ''))
                txn['reference'] = line['Transaction number']
                txn['sender'] = line['???transfer.fromOwner???']
                txn['recipient'] = line['???transfer.toOwner???']
                txn['currency'] = 'TSH'
                txn['comment'] = line['Transaction type']

                transactions.append(txn)

        return transactions