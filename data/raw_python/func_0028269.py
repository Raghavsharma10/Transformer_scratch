def get_auth(self):
        """
        This method requests an authentication object from the HPE IMC NMS and returns an
        HTTPDigest Auth Object
        :return:
        """
        url = self.h_url + self.server + ":" + self.port
        auth = requests.auth.HTTPDigestAuth(self.username, self.password)
        f_url = url + "/imcrs"
        try:
            response = requests.get(f_url, auth=auth, headers=HEADERS, verify=False)
            if response.status_code != 200:  # checks for valid IMC credentials
                print("Error:\n" + "Error: \n You're credentials are invalid. Please try again\n\n")
                set_imc_creds()
            return response.status_code
        except requests.exceptions.RequestException as error:
            return "Error:\n" + str(error) + 'The IMC server address is invalid. Please try again'