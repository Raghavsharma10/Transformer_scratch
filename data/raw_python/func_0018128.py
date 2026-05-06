def _parse_and_format(self):
        """ Parse and format the results returned by the ACS Zeropoint Calculator.

        Using ``beautifulsoup4``, find all the ``<tb> </tb>`` tags present in
        the response. Format the results into an astropy.table.QTable with
        corresponding units and assign it to the zpt_table attribute.
        """

        soup = BeautifulSoup(self._response.read(), 'html.parser')

        # Grab all elements in the table returned by the ZPT calc.
        td = soup.find_all('td')

        # Remove the units attached to PHOTFLAM and PHOTPLAM column names.
        td = [val.text.split(' ')[0] for val in td]

        # Turn the single list into a 2-D numpy array
        data = np.reshape(td,
                          (int(len(td) / self._block_size), self._block_size))
        # Create the QTable, note that sometimes self._response will be empty
        # even though the return was successful; hence the try/except to catch
        # any potential index errors. Provide the user with a message and
        # set the zpt_table to None.
        try:
            tab = QTable(data[1:, :],
                         names=data[0],
                         dtype=[str, float, float, float, float, float])
        except IndexError as e:
            msg = ('{}\n{}\n There was an issue parsing the request. '
                   'Try resubmitting the query. If this issue persists, please '
                   'submit a ticket to the Help Desk at'
                   'https://stsci.service-now.com/hst'
                   .format(e, self._msg_div))
            LOG.info(msg)
            self._zpt_table = None
        else:
            # If and only if no exception was raised, attach the units to each
            # column of the QTable. Note we skip the FILTER column because
            # Quantity objects in astropy must be numerical (i.e. not str)
            for col in tab.colnames:
                if col.lower() == 'filter':
                    continue
                tab[col].unit = self._data_units[col]

            self._zpt_table = tab