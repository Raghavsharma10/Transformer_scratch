def parseLegacy(self, response):
        """
        Parse a legacy response and try and catch any errors. If we have multiple
        responses we wont catch any exceptions, we will return the errors
        row by row

        :param dict response: The response string returned from request()

        :return Returns a dictionary or a list (list for multiple responses)
        """
        lines = response.splitlines()
        result = []
        pattern = re.compile('([A-Za-z]+):((.(?![A-Za-z]+:))*)')

        for line in lines:
            matches = pattern.findall(line)
            row = {}

            for match in matches:
                row[match[0]] = match[1].strip()

            try:
                error = row['ERR'].split(',')
            except KeyError:
                pass
            else:
                row['code'] = error[0] if len(error) == 2 else 0
                row['error'] = error[1].strip() if len(error) == 2 else error[0]
                del row['ERR']

                # If this response is a single row response, then we will throw
                # an exception to alert the user of any failures.
                if (len(lines) == 1):
                    raise ClickatellError(row['error'], row['code'])
            finally:
                result.append(row)

        return result if len(result) > 1 else result[0]