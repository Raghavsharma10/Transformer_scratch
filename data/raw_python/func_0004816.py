def _extract_values(values_list):
        """extract values from either file or list

        :param values_list: list or file name (str) with list of values
        """
        values = []
        # check if file or list of values to iterate
        if isinstance(values_list, str):
            with open(values_list) as ff:
                reading = csv.reader(ff)
                for j in reading:
                    values.append(j[0])
        elif isinstance(values_list, list):
            values = values_list
        else:
            raise Exception("input datatype not supported.")
        return values