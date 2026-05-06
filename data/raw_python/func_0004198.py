def list_all_by_reqvip(self, id_vip, pagination):
        """
            List All Pools To Populate Datatable

            :param pagination: Object Pagination

            :return: Following dictionary:{
                                            "total" : < total >,
                                            "pools" :[{
                                                "id": < id >
                                                "default_port": < default_port >,
                                                "identifier": < identifier >,
                                                "healthcheck": < healthcheck >,
                                            }, ... too ... ]}

            :raise NetworkAPIException: Falha ao acessar fonte de dados
        """

        uri = "api/pools/pool_list_by_reqvip/"

        data = dict()

        data["start_record"] = pagination.start_record
        data["end_record"] = pagination.end_record
        data["asorting_cols"] = pagination.asorting_cols
        data["searchable_columns"] = pagination.searchable_columns
        data["custom_search"] = pagination.custom_search or None
        data["id_vip"] = id_vip or None

        return self.post(uri, data=data)