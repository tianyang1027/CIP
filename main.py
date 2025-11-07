import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from document_parser import extract_steps_from_left_pane, extract_steps_from_right_pane
from image_to_steps_check import compare_operations

def main():
    driver = webdriver.Edge()
    driver.get("https://crowdintelligence.azurewebsites.net/triaging/v1/permalink?metric=GenericScenario&auditid=41722&build=25100503323261&hitid=385734864&judgeid=2992258")
    
    sign_in_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
    )

    sign_in_button.click()
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "leftPane"))
    )
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col"))
    )
    standard_steps = extract_steps_from_left_pane(driver)
    actual_steps = extract_steps_from_right_pane(driver)
    driver.quit()
    report = compare_operations(standard_steps, actual_steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
