from selenium.webdriver.common.by import By
import json
import time

def extract_steps_from_left_pane(driver, wait_time=2):
    time.sleep(wait_time)
    lis = driver.find_elements(By.XPATH, "//*[@id='leftPane']//li")
    
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
    iframe = driver.find_element(By.CSS_SELECTOR, "#judge-comment iframe")

    driver.switch_to.frame(iframe)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

    li_elements = driver.find_elements(By.TAG_NAME, "li")
    
    result = []
    for li in li_elements:
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