def comparison_table(self, caption=None, label="tab:model_comp", hlines=True,
                         aic=True, bic=True, dic=True, sort="bic", descending=True):  # pragma: no cover
        """
        Return a LaTeX ready table of model comparisons.

        Parameters
        ----------
        caption : str, optional
            The table caption to insert.
        label : str, optional
            The table label to insert.
        hlines : bool, optional
            Whether to insert hlines in the table or not.
        aic : bool, optional
            Whether to include a column for AICc or not.
        bic : bool, optional
            Whether to include a column for BIC or not.
        dic : bool, optional
            Whether to include a column for DIC or not.
        sort : str, optional
            How to sort the models. Should be one of "bic", "aic" or "dic".
        descending : bool, optional
            The sort order.

        Returns
        -------
        str
            A LaTeX table to be copied into your document.
        """

        if sort == "bic":
            assert bic, "You cannot sort by BIC if you turn it off"
        if sort == "aic":
            assert aic, "You cannot sort by AIC if you turn it off"
        if sort == "dic":
            assert dic, "You cannot sort by DIC if you turn it off"

        if caption is None:
            caption = ""
        if label is None:
            label = ""

        base_string = get_latex_table_frame(caption, label)
        end_text = " \\\\ \n"
        num_cols = 1 + (1 if aic else 0) + (1 if bic else 0)
        column_text = "c" * (num_cols + 1)
        center_text = ""
        hline_text = "\\hline\n"
        if hlines:
            center_text += hline_text
        center_text += "\tModel" + (" & AIC" if aic else "") + (" & BIC " if bic else "") \
                       + (" & DIC " if dic else "") + end_text
        if hlines:
            center_text += "\t" + hline_text
        if aic:
            aics = self.aic()
        else:
            aics = np.zeros(len(self.parent.chains))
        if bic:
            bics = self.bic()
        else:
            bics = np.zeros(len(self.parent.chains))
        if dic:
            dics = self.dic()
        else:
            dics = np.zeros(len(self.parent.chains))

        if sort == "bic":
            to_sort = bics
        elif sort == "aic":
            to_sort = aics
        elif sort == "dic":
            to_sort = dics
        else:
            raise ValueError("sort %s not recognised, must be dic, aic or dic" % sort)

        good = [i for i, t in enumerate(to_sort) if t is not None]
        names = [self.parent.chains[g].name for g in good]
        aics = [aics[g] for g in good]
        bics = [bics[g] for g in good]
        to_sort = bics if sort == "bic" else aics

        indexes = np.argsort(to_sort)

        if descending:
            indexes = indexes[::-1]

        for i in indexes:
            line = "\t" + names[i]
            if aic:
                line += "  &  %5.1f  " % aics[i]
            if bic:
                line += "  &  %5.1f  " % bics[i]
            if dic:
                line += "  &  %5.1f  " % dics[i]
            line += end_text
            center_text += line
        if hlines:
            center_text += "\t" + hline_text

        return base_string % (column_text, center_text)