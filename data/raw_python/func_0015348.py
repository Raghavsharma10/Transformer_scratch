def list_view_row_clicked(self, list_view, path, view_column):
        """
        Function opens the firefox window with relevant link
        """
        model = list_view.get_model()
        text = model[path][0]
        match = URL_FINDER.search(text)
        if match is not None:
            url = match.group(1)
            import webbrowser

            webbrowser.open(url)