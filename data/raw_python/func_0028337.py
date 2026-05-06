def get_auth(self):
        """
        This method requests an authentication object from the HPE IMC NMS and returns an HTTPDigest Auth Object
        :return:
        """
        url = self.h_url + self.server + ":" + self.port
        auth = requests.auth.HTTPDigestAuth(self.username,self.password)
        auth_url = "/imcrs"
        f_url = url + auth_url
        try:
            r = requests.get(f_url, auth=auth, headers=headers, verify=False)
            return r.status_code
    # checks for reqeusts exceptions
        except requests.exceptions.RequestException as e:
            return ("Error:\n" + str(e) + '\n\nThe IMC server address is invalid. Please try again')
            set_imc_creds()
        if r.status_code != 200:  # checks for valid IMC credentials
            return ("Error:\n" + str(e) +"Error: \n You're credentials are invalid. Please try again\n\n")
            set_imc_creds()