import sys
import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from document_parser import extract_steps_from_left_pane, extract_steps_from_right_pane
from image_to_steps_check import compare_operations

def main(page_url: str):
    # Initialize the Edge WebDriver
    driver = webdriver.Edge()
    
    # Open the given page URL
    driver.get(page_url)
    driver.maximize_window()
    
    # Wait for the sign-in button to be clickable
    sign_in_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
    )
    sign_in_button.click()

    # Wait until the left and right panes are loaded
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "leftPane"))
    )
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col"))
    )

    # Extract standard steps and actual steps from the page
    standard_steps = extract_steps_from_left_pane(driver)
    print(f"Standard steps extracted: {len(standard_steps)}")
    actual_steps = extract_steps_from_right_pane(driver)
    print(f"Actual steps extracted: {len(actual_steps)}")

    # Validate extracted steps
    if not standard_steps:
        print("Error: No standard steps found in the left pane.")
        driver.quit()
        return

    if not actual_steps:
        print("Error: No actual steps found in the right pane.")
        driver.quit()
        return

    if len(standard_steps) != len(actual_steps):
        print("Error: Mismatched number of steps.")
        driver.quit()
        return
    
    print("All checks passed.")
    # Close the browser
    driver.quit()

    print("Comparing steps...")

    # Compare the steps and print the report in JSON format
    report = compare_operations(standard_steps, actual_steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Ensure the file is saved in the root directory
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # output_path = os.path.join(root_dir, "report.txt")

    # Write the JSON report to the file
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(report, f, ensure_ascii=False, indent=2)
    
    # print(f"Report saved to {output_path}")

if __name__ == "__main__":
    # Ensure the user provides a page URL
    if len(sys.argv) < 2:
        print("Please provide a page URL:")
        sys.exit(1)
    
    page_url = sys.argv[1]
    print(f"Starting check for URL: {page_url}")
    main(page_url)
