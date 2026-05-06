def create_url(self, db="", user="genome", host="genome-mysql.cse.ucsc.edu",
        password="", dialect="mysqldb"):
        """
        internal: create a dburl from a set of parameters or the defaults on
        this object
        """
        if os.path.exists(db):
            db = "sqlite:///" + db

        # Is this a DB URL? If so, use it directly
        if self.db_regex.match(db):
            self.db = self.url = db
            self.dburl = db
            self.user = self.host = self.password = ""
        else:
            self.db = db
            if user == "genome" and host != "genome-mysql.cse.ucsc.edu":
                import getpass
                user = getpass.getuser()
            self.host = host
            self.user = user
            self.password = (":" + password) if password else ""
            self.dburl = self.url.format(db=self.db, user=self.user,
                host=self.host, password=self.password, dialect=dialect)