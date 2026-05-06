def get_people(self, user_alias=None):
        """
        获取用户信息
        
        :param user_alias: 用户ID
        :return: 
        """
        user_alias = user_alias or self.api.user_alias
        content = self.api.req(API_PEOPLE_HOME % user_alias).content
        xml = self.api.to_xml(re.sub(b'<br />', b'\n', content))
        try:
            xml_user = xml.xpath('//*[@id="profile"]')
            if not xml_user:
                return None
            else:
                xml_user = xml_user[0]
            avatar = first(xml_user.xpath('.//img/@src'))
            city = first(xml_user.xpath('.//div[@class="user-info"]/a/text()'))
            city_url = first(xml_user.xpath('.//div[@class="user-info"]/a/@href'))
            text_created_at = xml_user.xpath('.//div[@class="pl"]/text()')[1]
            created_at = re.match(r'.+(?=加入)', text_created_at.strip()).group()
            xml_intro = first(xml.xpath('//*[@id="intro_display"]'))
            intro = xml_intro.xpath('string(.)') if xml_intro is not None else None
            nickname = first(xml.xpath('//*[@id="db-usr-profile"]//h1/text()'), '').strip() or None
            signature = first(xml.xpath('//*[@id="display"]/text()'))
            xml_contact_count = xml.xpath('//*[@id="friend"]/h2')[0]
            contact_count = int(re.search(r'成员(\d+)', xml_contact_count.xpath('string(.)')).groups()[0])
            text_rev_contact_count = xml.xpath('//p[@class="rev-link"]/a/text()')[0]
            rev_contact_count = int(re.search(r'(\d+)人关注', text_rev_contact_count.strip()).groups()[0])
            return {
                'alias': user_alias,
                'url': API_PEOPLE_HOME % user_alias,
                'avatar': avatar,
                'city': city,
                'city_url': city_url,
                'created_at': created_at,
                'intro': intro,
                'nickname': nickname,
                'signature': signature,
                'contact_count': contact_count,
                'rev_contact_count': rev_contact_count,
            }
        except Exception as e:
            self.api.logger.exception('parse people meta error: %s' % e)