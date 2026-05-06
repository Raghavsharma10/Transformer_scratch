def get_date_datetime_param(self, request, param):
        """Check the request for the provided query parameter and returns a rounded value.

        :param request: WSGI request object to retrieve query parameter data.
        :param param: the name of the query parameter.
        """
        if param in request.GET:
            param_value = request.GET.get(param, None)
            # Match and interpret param if formatted as a date.
            date_match = dateparse.date_re.match(param_value)
            if date_match:
                return timezone.datetime.combine(
                    dateparse.parse_date(date_match.group(0)), timezone.datetime.min.time()
                )
            datetime_match = dateparse.datetime_re.match(param_value)
            if datetime_match:
                return timezone.datetime.combine(
                    dateparse.parse_datetime(datetime_match.group(0)).date(),
                    timezone.datetime.min.time()
                )
        return None