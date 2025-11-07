import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from document_parser import extract_steps_from_left_pane, extract_steps_from_right_pane
from image_to_steps_check import compare_operations

def main():
    driver = webdriver.Edge()
    driver.get("https://crowdintelligence.azurewebsites.net/triaging/v1/permalink?metric=GenericScenario&auditid=41722&build=25100503323261&hitid=385734864&judgeid=2992258")
    standard_steps = extract_steps_from_left_pane(driver)
    actual_steps = extract_steps_from_right_pane(driver)
    driver.quit()
    report = compare_operations(standard_steps, actual_steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
