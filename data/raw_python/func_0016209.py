def get_package_id(self, name):
        """
        Retrieve the smart package id given is English name
        @param (str) name: the Aruba Smart package size name, ie: "small", "medium", "large", "extra large".
        @return: The package id that depends on the Data center and the size choosen.
        """
        json_scheme = self.gen_def_json_scheme('GetPreConfiguredPackages', dict(HypervisorType=4))
        json_obj = self.call_method_post(method='GetPreConfiguredPackages ', json_scheme=json_scheme)
        for package in json_obj['Value']:
            packageId  = package['PackageID']
            for description in package['Descriptions']:
                languageID = description['LanguageID']
                packageName = description['Text']
                if languageID == 2 and packageName.lower() == name.lower():
                    return packageId