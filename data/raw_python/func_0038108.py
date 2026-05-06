def deregister(self, reg_data, retry=True, interval=1, timeout=3):
        """
        Deregister model/view of this bundle
        """
        Retry(target=self.publish.direct.delete,
              args=("/controller/registration", reg_data,),
              kwargs={"timeout": timeout},
              options={"retry": retry, "interval": interval})
        _logger.debug("Deregister successfully %s tunnel: %s" %
                      (reg_data["name"],
                       self._conn.tunnels[reg_data["role"]][0],))