def __enableProxy(self):
        """
        Set the required environment variables to enable the use of hoverfly as a proxy.
        """
        os.environ[
            "HTTP_PROXY"] = self.httpProxy()
        os.environ[
            "HTTPS_PROXY"] = self.httpsProxy()

        os.environ["REQUESTS_CA_BUNDLE"] = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "cert.pem")