def release_data(self, package_name, version):
        """Query PYPI via XMLRPC interface for a pkg's metadata"""
        try:
            return self.xmlrpc.release_data(package_name, version)
        except xmlrpclib.Fault:
            #XXX Raises xmlrpclib.Fault if you give non-existant version
            #Could this be server bug?
            return