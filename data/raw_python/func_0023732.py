def _get_torrent(self, row):
        """
        Parse row into namedtuple
        """
        td = row("td")
        name = td("a.cellMainLink").text()
        name = name.replace(" . ", ".").replace(" .", ".")
        author = td("a.plain").text()
        verified_author = True if td(".lightgrey>.ka-verify") else False
        category = td("span").find("strong").find("a").eq(0).text()
        verified_torrent = True if td(".icon16>.ka-green") else False
        comments = td(".iaconbox>.icommentjs>.iconvalue").text()
        torrent_link = "http://" + BASE.domain
        if td("a.cellMainLink").attr("href") is not None:
            torrent_link += td("a.cellMainLink").attr("href")
        magnet_link = td("a[data-nop]").eq(1).attr("href")
        download_link = td("a[data-download]").attr("href")

        td_centers = row("td.center")
        size = td_centers.eq(0).text()
        files = td_centers.eq(1).text()
        age = " ".join(td_centers.eq(2).text().split())
        seed = td_centers.eq(3).text()
        leech = td_centers.eq(4).text()

        return Torrent(name, author, verified_author, category, size,
                       files, age, seed, leech, verified_torrent, comments,
                       torrent_link, magnet_link, download_link)