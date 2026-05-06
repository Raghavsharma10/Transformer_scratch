def write_response_html_to_file(response,filename):
        """
        An aid in troubleshooting internal application errors, i.e.  <Response [500]>, to be mainly
        beneficial when developing the server-side API. This method will write the response HTML
        for viewing the error details in the browesr.

        Args:
            response: `requests.models.Response` instance.
            filename: `str`. The output file name.
        """
        fout = open(filename,'w')
        if not str(response.status_code).startswith("2"):
            Model.debug_logger.debug(response.text)
        fout.write(response.text)
        fout.close()