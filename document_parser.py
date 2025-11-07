from selenium.webdriver.common.by import By
import json
import time

def extract_steps_from_left_pane(driver, wait_time=2):
    time.sleep(wait_time)
    lis = driver.find_elements(By.ID, "leftPane li")
    
    result = []
    for li in lis:
        try:
            img_tag = li.find_element(By.TAG_NAME, "img")
            img = img_tag.get_attribute("src")
        except:
            img = None

        text = li.text.strip()
        result.append({
            "text": text,
            "img": img
        })

    return result


def extract_steps_from_right_pane(driver, wait_time=2):
    time.sleep(wait_time)
    lis = driver.find_elements(By.CSS_SELECTOR, ".right-pane.col li")
    
    result = []
    for li in lis:
        try:
            img_tag = li.find_element(By.TAG_NAME, "img")
            img = img_tag.get_attribute("src")
        except:
            img = None
        
        p=li.find_element(By.TAG_NAME, "p")
        text = p.text.strip()
        result.append({
            "text": text,
            "img": img
        })

    return result