def build_layout(self, dset_id: str):
        """
        :param dset_id:
        :return:
        """
        all_fields = list(self.get_data(dset_id=dset_id).keys())

        try:
            field_reference = self.skd[dset_id].attrs('target')
        except:
            field_reference = all_fields[0]

        fields_comparison = [all_fields[1]]

        # chart type widget
        self.register_widget(
            chart_type=widgets.RadioButtons(
                options=['individual', 'grouped'],
                value='individual',
                description='Chart Type:'
            )
        )

        # bins widget
        self.register_widget(
            bins=IntSlider(
                description='Bins:',
                min=2, max=10, value=2,
                continuous_update=False
            )
        )

        # fields comparison widget
        self.register_widget(
            xs=widgets.SelectMultiple(
                description='Xs:',
                options=[f for f in all_fields if not f == field_reference],
                value=fields_comparison
            )
        )

        # field reference widget
        self.register_widget(
            y=widgets.Dropdown(
                description='Y:',
                options=all_fields,
                value=field_reference
            )
        )
        # used to internal flow control
        y_changed = [False]

        self.register_widget(
            box_filter_panel=widgets.VBox([
                self._('y'), self._('xs'), self._('bins')
            ])
        )

        # layout widgets
        self.register_widget(
            table=widgets.HTML(),
            chart=widgets.HTML()
        )

        self.register_widget(vbox_chart=widgets.VBox([
            self._('chart_type'), self._('chart')
        ]))

        self.register_widget(
            tab=widgets.Tab(
                children=[
                    self._('box_filter_panel'),
                    self._('table'),
                    self._('vbox_chart')
                ]
            )
        )

        self.register_widget(dashboard=widgets.HBox([self._('tab')]))

        # observe hooks
        def w_y_change(change: dict):
            """
            When y field was changed xs field should be updated and data table
            and chart should be displayed/updated.

            :param change:
            :return:
            """
            # remove reference field from the comparison field list
            _xs = [
                f for f in all_fields
                if not f == change['new']
            ]

            y_changed[0] = True  # flow control variable
            _xs_value = list(self._('xs').value)

            if change['new'] in self._('xs').value:
                _xs_value.pop(_xs_value.index(change['new']))
                if not _xs_value:
                    _xs_value = [_xs[0]]

            self._('xs').options = _xs
            self._('xs').value = _xs_value

            self._display_result(y=change['new'], dset_id=dset_id)

            y_changed[0] = False  # flow control variable

        # widgets registration

        # change tab settings
        self._('tab').set_title(0, 'Filter')
        self._('tab').set_title(1, 'Data')
        self._('tab').set_title(2, 'Chart')

        # data panel
        self._('table').value = '...'

        # chart panel
        self._('chart').value = '...'

        # create observe callbacks
        self._('bins').observe(
            lambda change: (
                self._display_result(bins=change['new'], dset_id=dset_id)
            ), 'value'
        )
        self._('y').observe(w_y_change, 'value')
        # execute display result if 'y' was not changing.
        self._('xs').observe(
            lambda change: (
                self._display_result(xs=change['new'], dset_id=dset_id)
                if not y_changed[0] else None
            ), 'value'
        )
        self._('chart_type').observe(
            lambda change: (
                self._display_result(chart_type=change['new'], dset_id=dset_id)
            ), 'value'
        )