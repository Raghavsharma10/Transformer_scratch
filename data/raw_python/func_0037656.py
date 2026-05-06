def _extract_proxies(self, ajax_endpoint):
    
        ''' request the xml object '''
        proxy_xml = requests.get(ajax_endpoint)
        print(proxy_xml.content)
        root = etree.XML(proxy_xml.content)
        quote = root.xpath('quote')[0]
        
        ''' extract the raw text from the body of the quote tag '''
        raw_text = quote.text
        
        ''' eliminate the stuff we don't need '''
        proxy_data = raw_text.split('You will definitely love it! Give it a try!</td></tr>')[1]
        
        ''' get rid of the </table> at the end of proxy_data '''
        proxy_data = proxy_data[:-len('</table>')]
        
        ''' split proxy_data into rows '''
        table_rows = proxy_data.split('<tr>')
        
        ''' convert each row into a Proxy object '''
        for row in table_rows:
            
            ''' get rid of the </tr> at the end of each row '''
            row = row[:-len('</tr>')]
        
            ''' split each row into a list of items '''
            items = row.split('<td>')
            
            ''' sometimes we get weird lists containing only an empty string '''
            if len(items) != 7:
                continue
    
            ''' we'll use this to remove the </td> from the end of each item '''
            tdlen = len('</td>')
    
            ''' create proxy dict '''
            proxy = Proxy(
                ip=items[1][:-tdlen],
                port=int(items[2][:-tdlen]),
                https=bool(items[3][:-tdlen]),
                latency=int(items[4][:-tdlen]),
                last_checked=items[5][:-tdlen],
                country=items[6][:-tdlen],
            )
            yield proxy