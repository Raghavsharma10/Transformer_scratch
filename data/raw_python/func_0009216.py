def __get_strut_token(self):
        """
        Move the staged loan notes to the order stage and get the struts token
        from the place order HTML.
        The order will not be placed until calling _confirm_order()

        Returns
        -------
        dict
            A dict with the token name and value
        """

        try:
            # Move to the place order page and get the struts token

            response = self.lc.session.get('/portfolio/placeOrder.action')
            soup = BeautifulSoup(response.text, "html5lib")


            # Example HTML with the stuts token:
            """
            <input type="hidden" name="struts.token.name" value="token" />
            <input type="hidden" name="token" value="C4MJZP39Q86KDX8KN8SBTVCP0WSFBXEL" />
            """
            # 'struts.token.name' defines the field name with the token value

            strut_tag = None
            strut_token_name = soup.find('input', {'name': 'struts.token.name'})
            if strut_token_name and strut_token_name['value'].strip():

                # Get form around the strut.token.name element
                form = soup.form # assumed
                for parent in strut_token_name.parents:
                    if parent and parent.name == 'form':
                        form = parent
                        break

                # Get strut token value
                strut_token_name = strut_token_name['value']
                strut_tag = soup.find('input', {'name': strut_token_name})
                if strut_tag and strut_tag['value'].strip():
                    return {'name': strut_token_name, 'value': strut_tag['value'].strip()}

            # No strut token found
            self.__log('No struts token! HTML: {0}'.format(response.text))
            raise LendingClubError('No struts token. Please report this error.', response)

        except Exception as e:
            self.__log('Could not get struts token. Error message: {0}'.format(str(e)))
            raise LendingClubError('Could not get struts token. Error message: {0}'.format(str(e)))