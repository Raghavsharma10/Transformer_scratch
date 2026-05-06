def find_logs(
            self,
            user_name,
            first_date,
            start_time,
            last_date,
            end_time,
            action,
            functionality,
            parameter,
            pagination):
        """
        Search all logs, filtering by the given parameters.
        :param user_name: Filter by user_name
        :param first_date: Sets initial date for begin of the filter
        :param start_time: Sets initial time
        :param last_date: Sets final date
        :param end_time: Sets final time and ends the filter. That defines the searching gap
        :param action: Filter by action (Create, Update or Delete)
        :param functionality: Filter by class
        :param parameter: Filter by parameter
        :param pagination: Class with all data needed to paginate

        :return: Following dictionary:

        ::

            {'eventlog': {'id_usuario' : < id_user >,
            'hora_evento': < hora_evento >,
            'acao': < acao >,
            'funcionalidade': < funcionalidade >,
            'parametro_anterior': < parametro_anterior >,
            'parametro_atual': < parametro_atual > }
            'total' : {< total_registros >} }

        :raise InvalidParameterError: Some parameter was invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not isinstance(pagination, Pagination):
            raise InvalidParameterError(
                u"Invalid parameter: pagination must be a class of type 'Pagination'.")

        eventlog_map = dict()

        eventlog_map["start_record"] = pagination.start_record
        eventlog_map["end_record"] = pagination.end_record
        eventlog_map["asorting_cols"] = pagination.asorting_cols
        eventlog_map["searchable_columns"] = pagination.searchable_columns
        eventlog_map["custom_search"] = pagination.custom_search

        eventlog_map["usuario"] = user_name
        eventlog_map["data_inicial"] = first_date
        eventlog_map["hora_inicial"] = start_time
        eventlog_map["data_final"] = last_date
        eventlog_map["hora_final"] = end_time
        eventlog_map["acao"] = action
        eventlog_map["funcionalidade"] = functionality
        eventlog_map["parametro"] = parameter

        url = "eventlog/find/"

        code, xml = self.submit({'eventlog': eventlog_map}, 'POST', url)

        key = "eventlog"
        return get_list_map(self.response(code, xml, key), key)