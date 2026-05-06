def SwitchToAlert(): 
        ''' <input value="Test" type="button" onClick="alert('OK')" > '''
        try:            
            alert = WebDriverWait(Web.driver, 10).until(lambda driver: driver.switch_to_alert())
            return alert            
        except:            
            print("Waring: Timeout at %d seconds.Alert was not found.")
            return False