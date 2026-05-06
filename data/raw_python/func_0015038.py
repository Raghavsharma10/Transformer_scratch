def _sort_and_trim(data, reverse=False):
        """Sorts a dictionary with at least two fields on each of them sorting
        by the second element.

        .. warning::
          Right now is hardcoded to 10 elements, improve the command line
          interface to allow to send parameters to each command or globally.
        """
        threshold = 10
        data_list = data.items()
        data_list = sorted(
            data_list,
            key=lambda data_info: data_info[1],
            reverse=reverse,
        )
        return data_list[:threshold]