def interactive_client_choice(self):
        """
        Present a menu for user to select from ESP/MSP managed clients they have permission to.

        **Returns:** Tuple with (Boolean success, selected client ID).
        """

        clients = self._parent_class.get.clients_t()
        clients_perms = self._parent_class.get.permissions_clients_d(self._parent_class._user_id)

        client_status = clients.cgx_status
        clients_dict = clients.cgx_content
        c_perms_status = clients_perms.cgx_status
        c_perms_dict = clients_perms.cgx_content

        # Build MSP/ESP id-name dict, get list of allowed tenants.
        if client_status and c_perms_status:
            client_id_name = {}
            for client in clients_dict.get('items', []):
                if type(client) is dict:
                    # create client ID to name map table.
                    client_id_name[client.get('id', "err")] = client.get('canonical_name')

            # Valid clients w/permissions - create list of tuples for menu
            menu_list = []
            for client in c_perms_dict.get('items', []):
                if type(client) is dict:
                    # add entry
                    client_id = client.get('client_id')
                    # create tuple of ( client name, client id ) to append to list
                    menu_list.append(
                        (client_id_name.get(client_id, client_id), client_id)
                    )
            # empty menu?
            if not menu_list:
                # no clients
                print("No ESP/MSP clients allowed for user.")
                return False, {}

            # ask user to select client
            _, chosen_client_id = self.quick_menu("ESP/MSP Detected. Select a client to use:", "{0}) {1}", menu_list)

            return True, chosen_client_id

        else:
            print("ESP/MSP detail retrieval failed.")
            return False, {}