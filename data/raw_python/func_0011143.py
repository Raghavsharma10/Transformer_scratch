def public_api(self,url):
        ''' template function of public api'''
        try :
            url in api_urls
            return ast.literal_eval(requests.get(base_url + api_urls.get(url)).text)
        except Exception as e:
            print(e)