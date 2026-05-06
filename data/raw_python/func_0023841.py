def create_server(self, server):
        """
        Create a server and its storages based on a (locally created) Server object.

        Populates the given Server instance with the API response.

        0.3.0: also supports giving the entire POST body as a dict that is directly
        serialised into JSON. Refer to the REST API documentation for correct format.

        Example:
        server1 = Server( core_number = 1,
              memory_amount = 1024,
              hostname = "my.example.1",
              zone = ZONE.London,
              storage_devices = [
                Storage(os = "Ubuntu 14.04", size=10, tier=maxiops, title='The OS drive'),
                Storage(size=10),
                Storage()
              title = "My Example Server"
            ])
        manager.create_server(server1)

        One storage should contain an OS. Otherwise storage fields are optional.
        - size defaults to 10,
        - title defaults to hostname + " OS disk" and hostname + " storage disk id"
          (id is a running starting from 1)
        - tier defaults to maxiops
        - valid operating systems are:
          "CentOS 6.5", "CentOS 7.0"
          "Debian 7.8"
          "Ubuntu 12.04", "Ubuntu 14.04"
          "Windows 2003","Windows 2008" ,"Windows 2012"
        """
        if isinstance(server, Server):
            body = server.prepare_post_body()
        else:
            server = Server._create_server_obj(server, cloud_manager=self)
            body = server.prepare_post_body()

        res = self.post_request('/server', body)

        server_to_return = server
        server_to_return._reset(
            res['server'],
            cloud_manager=self,
            populated=True
        )
        return server_to_return