def getRealContributions(self):
        """Get the real number of contributions (private + public)."""
        datefrom = datetime.now() - relativedelta(days=366)
        dateto = datefrom + relativedelta(months=1) - relativedelta(days=1)
        private = 0

        while datefrom < datetime.now():
            fromstr = datefrom.strftime("%Y-%m-%d")
            tostr = dateto.strftime("%Y-%m-%d")
            url = self.server + self.name
            url += "?tab=overview&from=" + fromstr + "&to=" + tostr

            data = GitHubUser.__getDataFromURL(url)
            web = BeautifulSoup(data, "lxml")

            aux = "f4 lh-condensed m-0 text-gray"
            pcontribs = web.find_all("span", {"class": aux})

            aux = web.find_all('span', {'class': 'text-gray m-0'})

            noContribs = False

            for compr in aux:
                if "had no activity during this period." in compr.text:
                    noContribs = True

            try:
                if not noContribs:
                    for contrib in pcontribs:
                        contribution = None
                        contribution = contrib.text
                        contribution = contribution.lstrip().replace(",", "")
                        contribution = contribution.replace("\n", " ")
                        contribution = contribution.partition(" ")[0]
                        private += int(contribution)
            except IndexError as error:
                print("There was an error with the user " + self.name)
                print(error)
            except AttributeError as error:
                print("There was an error with the user " + self.name)
                print(error)

            datefrom += relativedelta(months=1)
            dateto += relativedelta(months=1)

        self.private = private
        self.public = self.contributions - private

        if self.public < 0:  # Is not exact
            self.public = 0