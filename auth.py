from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


driver = webdriver.Edge()
driver.get("https://crowdintelligence.azurewebsites.net/audit")

sign_in_button = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
)

sign_in_button.click()

dropdown = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.ID, "Dropdown12"))
)

dropdown.click()

option = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, "Dropdown12-list1"))
)
option.click()

table = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.TAG_NAME, "table"))
)

rows = table.find_elements(By.TAG_NAME, "tr")

for i, row in enumerate(rows):
    try:
        buttons = row.find_elements(By.TAG_NAME, "button")
        if buttons:
            last_button = buttons[-1]
            driver.execute_script("arguments[0].scrollIntoView(true);", last_button)
            last_button.click()
            print(f"✅ 点击第 {i+1} 行的按钮成功")
            time.sleep(1) 
    except Exception as e:
        print(f"❌ 点击第 {i+1} 行按钮失败：{e}")

print("操作完成！")
driver.quit()
