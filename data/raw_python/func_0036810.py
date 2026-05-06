def component(self, *components):
        r"""
            When search() is called it will limit results to items in a component.

            :param component: items passed in will be turned into a list
            :returns: :class:`Search`
        """
        for component in components:
            self._component.append(component)
        return self