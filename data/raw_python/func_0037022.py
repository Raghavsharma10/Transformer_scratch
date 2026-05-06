async def async_set_port_poe_mode(self, port_idx, mode):
        """Set port poe mode.

        Auto, 24v, passthrough, off.
        Make sure to not overwrite any existing configs.
        """
        no_existing_config = True
        for port_override in self.port_overrides:
            if port_idx == port_override['port_idx']:
                port_override['poe_mode'] = mode
                no_existing_config = False
                break

        if no_existing_config:
            self.port_overrides.append({
                'port_idx': port_idx,
                'portconf_id': self.ports[port_idx].portconf_id,
                'poe_mode': mode
            })

        url = 's/{site}/rest/device/' + self.id
        data = {'port_overrides': self.port_overrides}

        await self._request('put', url, json=data)