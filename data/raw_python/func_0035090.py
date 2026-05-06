def _set_label(self, which, label, **kwargs):
        """Private method for setting labels.

        Args:
            which (str): The indicator of which part of the plots
                to adjust. This currently handles `xlabel`/`ylabel`,
                and `title`.
            label (str): The label to be added.
            fontsize (int, optional): Fontsize for associated label. Default
                is None.

        """
        prop_default = {
            'fontsize': 18,
        }

        for prop, default in prop_default.items():
            kwargs[prop] = kwargs.get(prop, default)

        setattr(self.label, which, label)
        setattr(self.label, which + '_kwargs', kwargs)
        return