def _apply_udfs(self, record, hist, udf_type):
        """
        Excute user define processes, user-defined functionalty is designed to
        applyies custome trasformations to data.

        :param dict record: dictionary of values to validate
        :param dict hist: existing input of history values
        """

        def function_executor(func, *args):
            """
            Execute user define function
            :param python method func: Function obj
            :param methods arguments args: Function arguments
            """

            result, result_hist = func(*args)

            return result, result_hist

        if udf_type in self.udfs:

            cust_function_od_obj = collections.OrderedDict(
                sorted(
                    self.udfs[udf_type].items()
                )
            )

            for cust_function in cust_function_od_obj:

                record, hist = function_executor(
                    cust_function_od_obj[cust_function],
                    record,
                    hist
                )

        return record, hist