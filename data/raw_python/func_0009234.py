def load(self):
        """
        Load the filter from the server
        """

        # Attempt to load the saved filter
        payload = {
            'id': self.id
        }
        response = self.lc.session.get('/browse/getSavedFilterAj.action', query=payload)
        self.response = response
        json_response = response.json()

        if self.lc.session.json_success(json_response) and json_response['filterName'] != 'No filters':
            self.name = json_response['filterName']

            #
            # Parse out the filter JSON string manually from the response JSON.
            # If the filter JSON is modified at all, or any value is out of order,
            # LendingClub will reject the filter and perform a wildcard search instead,
            # without any error. So we need to retain the filter JSON value exactly how it is given to us.
            #
            text = response.text

            # Cut off everything  before "filter": [...]
            text = re.sub('\n', '', text)
            text = re.sub('^.*?,\s*["\']filter["\']:\s*\[(.*)', '[\\1', text)

            # Now loop through the string until we find the end of the filter block
            # This is a simple parser that keeps track of block elements, quotes and
            # escape characters
            blockTracker = []
            blockChars = {
                '[': ']',
                '{': '}'
            }
            inQuote = False
            lastChar = None
            json_text = ""
            for char in text:
                json_text += char

                # Escape char
                if char == '\\':
                    if lastChar == '\\':
                        lastChar = ''
                    else:
                        lastChar = char
                    continue

                # Quotes
                if char == "'" or char == '"':
                    if inQuote is False:  # Starting a quote block
                        inQuote = char
                    elif inQuote == char:  # Ending a quote block
                        inQuote = False
                    lastChar = char
                    continue

                # Start of a block
                if char in blockChars.keys():
                    blockTracker.insert(0, blockChars[char])

                # End of a block, remove from block path
                elif len(blockTracker) > 0 and char == blockTracker[0]:
                    blockTracker.pop(0)

                # No more blocks in the tracker which means we're at the end of the filter block
                if len(blockTracker) == 0 and lastChar is not None:
                    break

                lastChar = char

            # Verify valid JSON
            try:
                if json_text.strip() == '':
                    raise SavedFilterError('A saved filter could not be found for ID {0}'.format(self.id), response)

                json_test = json.loads(json_text)

                # Make sure it looks right
                assert type(json_test) is list, 'Expecting a list, instead received a {0}'.format(type(json_test))
                assert 'm_id' in json_test[0], 'Expecting a \'m_id\' property in each filter'
                assert 'm_value' in json_test[0], 'Expecting a \'m_value\' property in each filter'

                self.json = json_test
            except Exception as e:
                raise SavedFilterError('Could not parse filter from the JSON response: {0}'.format(str(e)))

            self.json_text = json_text
            self.__analyze()

        else:
            raise SavedFilterError('A saved filter could not be found for ID {0}'.format(self.id), response)