def get_selected_subassistant_path(self, **kwargs):
        """Recursively searches self._tree - has format of (Assistant: [list_of_subassistants]) -
        for specific path from first to last selected subassistants.

        Args:
            kwargs: arguments containing names of the given assistants in form of
            subassistant_0 = 'name', subassistant_1 = 'another_name', ...
        Returns:
            list of subassistants objects from tree sorted from first to last
        """
        path = [self]
        previous_subas_list = None
        currently_searching = self.get_subassistant_tree()[1]

        # len(path) - 1 always points to next subassistant_N, so we can use it to control iteration
        while settings.SUBASSISTANT_N_STRING.format(len(path) - 1) in kwargs and \
                kwargs[settings.SUBASSISTANT_N_STRING.format(len(path) - 1)]:
            for sa, subas_list in currently_searching:
                if sa.name == kwargs[settings.SUBASSISTANT_N_STRING.format(len(path) - 1)]:
                    currently_searching = subas_list
                    path.append(sa)
                    break  # sorry if you shed a tear ;)

            if subas_list == previous_subas_list:
                raise exceptions.AssistantNotFoundException(
                    'No assistant {n} after path {p}.'.format(
                        n=kwargs[settings.SUBASSISTANT_N_STRING.format(len(path) - 1)],
                        p=path))
            previous_subas_list = subas_list

        return path