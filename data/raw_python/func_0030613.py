def load(self, country="all"):
        """
        Load data
        """
        u = ("https://api.worldbank.org/v2/countries/{}/indicators/CPTOTSAXN"
             "?format=json&per_page=10000").format(country)
        r = requests.get(u)
        j = r.json()
        cpi_data = j[1]

        # Loop through the rows of the datapackage with the help of data
        for row in cpi_data:
            # Get the code and the name and transform to uppercase
            # so that it'll match no matter the case
            iso_3 = row["countryiso3code"].upper()
            iso_2 = row["country"]["id"].upper()
            name = row["country"]['value'].upper()
            # Get the date (which is in the field Year) and the CPI value
            date = row['date']
            cpi = row['value']
            for key in [iso_3, iso_2, name]:
                existing = self.data.get(key, {})
                existing[str(date)] = cpi
                if key:
                    self.data[key] = existing