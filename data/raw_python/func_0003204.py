def generate_report(self, template, path=None, style=None):
        """
         Generate HTML report

        Parameters
        ----------
        template : markdown-formatted string or path to the template
            file used for rendering the report. Any attribute of this
            object can be included in the report using the {tag} format.
            e.g.'# Report{estimator_name}{roc}{precision_recall}'.
            Apart from every attribute, you can also use {date} and {date_utc}
            tags to include the date for the report generation using local
            and UTC timezones repectively.

        path : str
            Path to save the HTML report. If None, the function will return
            the HTML code.

        style: str
            Path to a css file to apply style to the report. If None, no
            style will be applied

        Returns
        -------
        report: str
            Returns the contents of the report if path is None.

        """
        from .report import generate

        return generate(self, template, path, style)