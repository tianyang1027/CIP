import streamlit as st
import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from document_parser import extract_steps_from_left_pane, extract_steps_from_right_pane
from image_to_steps_check import compare_operations


def run_check(page_url: str):
    """Runs the Selenium automation and comparison logic."""
    st.info("🚀 Launching browser and opening the page, please wait...")

    try:
        # Initialize Edge WebDriver
        driver = webdriver.Edge()
        driver.get(page_url)
        driver.maximize_window()

        # Wait for the sign-in button to be clickable and click it
        sign_in_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
        )
        sign_in_button.click()

        # Wait for both left and right panes to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "leftPane"))
        )
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "right-pane.col"))
        )

        # Extract standard and actual steps from the web page
        standard_steps = extract_steps_from_left_pane(driver)
        actual_steps = extract_steps_from_right_pane(driver)

        # Close the browser
        driver.quit()

        # Validation checks
        if not standard_steps:
            st.error("❌ No standard steps found in the left pane.")
            return
        if not actual_steps:
            st.error("❌ No actual steps found in the right pane.")
            return
        if len(standard_steps) != len(actual_steps):
            st.error("❌ Mismatch between the number of standard and actual steps.")
            return

        st.success("✅ Steps extracted successfully. Starting comparison...")

        # Compare extracted steps
        report = compare_operations(standard_steps, actual_steps)

        # Display the report in JSON format
        st.json(report)

    except Exception as e:
        st.error(f"An error occurred: {e}")


def main():
    """Streamlit app entry point."""
    st.title("🔍 Web Steps Comparison Tool (Selenium + Streamlit)")
    st.write("Enter a webpage URL to automatically log in, extract, and compare the left and right pane steps.")

    # Input field for page URL
    page_url = st.text_input("Enter the page URL:")

    # Button to start the check
    if st.button("Start Checking"):
        if page_url.strip():
            run_check(page_url.strip())
        else:
            st.warning("Please enter a valid URL.")


if __name__ == "__main__":
    main()
