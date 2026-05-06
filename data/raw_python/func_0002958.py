def make_article_info_dates(self):
        """
        Makes the section containing important dates for the article: typically
        Received, Accepted, and Published.
        """
        dates_div = etree.Element('div', {'id': 'article-dates'})

        d = './front/article-meta/history/date'
        received = self.article.root.xpath(d + "[@date-type='received']")
        accepted = self.article.root.xpath(d + "[@date-type='accepted']")
        if received:
            b = etree.SubElement(dates_div, 'b')
            b.text = 'Received: '
            dt = self.date_tuple_from_date(received[0], 'Received')
            formatted_date_string = self.format_date_string(dt)
            append_new_text(dates_div, formatted_date_string + '; ')
        if accepted:
            b = etree.SubElement(dates_div, 'b')
            b.text = 'Accepted: '
            dt = self.date_tuple_from_date(accepted[0], 'Accepted')
            formatted_date_string = self.format_date_string(dt)
            append_new_text(dates_div, formatted_date_string + '; ')
        #Published date is required
        pub_date = self.article.root.xpath("./front/article-meta/pub-date[@pub-type='epub']")[0]
        b = etree.SubElement(dates_div, 'b')
        b.text = 'Published: '
        dt = self.date_tuple_from_date(pub_date, 'Published')
        formatted_date_string = self.format_date_string(dt)
        append_new_text(dates_div, formatted_date_string)

        return dates_div