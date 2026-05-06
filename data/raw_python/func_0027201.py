def display(self, dset_id: str):
        """
        :param dset_id:
        :return:

        """
        # update result
        self.skd[dset_id].compute()
        # build layout
        self.build_layout(dset_id=dset_id)
        # display widgets
        display(self._('dashboard'))
        # display data table and chart
        self._display_result(dset_id=dset_id)