import sys
import json
import os
import re
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)

from concurrent.futures import ThreadPoolExecutor

from parsers.document_parser import extract_steps_from_left_pane
# from parsers.document_parser import extract_steps_from_right_pane
# from llm.image_to_steps_check import compare_operations


# =======================
# 处理单个页面
# =======================
def process_page(page_url: str):
    if page_url is None:
        print("[SKIP] URL is None")
        return None
    page_url = page_url.strip()
    if not page_url or not page_url.startswith("https://"):
        print(f"[SKIP] Invalid URL: {page_url}")
        return None


    driver = None

    try:
        driver = webdriver.Edge()
        driver.set_page_load_timeout(20)

        # 打开页面（打不开会抛异常）
        driver.get(page_url)
        driver.maximize_window()

        # 等待登录按钮
        sign_in_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "signInColor"))
        )
        sign_in_button.click()

        # 等待左侧面板加载
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "leftPane"))
        )

        # 提取步骤
        standard_steps = extract_steps_from_left_pane(driver)
        print(f"[OK] {page_url} -> {len(standard_steps)} steps")

        return standard_steps

    except (TimeoutException, NoSuchElementException) as e:
        print(f"[SKIP] Page structure error: {page_url} | {e}")

    except WebDriverException as e:
        print(f"[SKIP] Page load failed: {page_url} | {e}")

    except Exception as e:
        print(f"[SKIP] Unknown error: {page_url} | {e}")

    finally:
        if driver:
            driver.quit()

    # 任何异常统一跳过
    return None


# =======================
# 处理 Excel
# =======================
def process_excel(file_path: str):
    df = pd.read_excel(file_path, engine="openpyxl")

    if "permalink" not in df.columns:
        print("Excel 中未找到 permalink 列")
        return

    links = df["permalink"].dropna().tolist()

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_page, url): url for url in links}

        for future in futures:
            page_url = futures[future]
            try:
                standard_steps = future.result()

                # 关键：None 直接跳过
                if not standard_steps:
                    continue

                results.append({
                    "page_url": page_url.strip(),
                    "standard_steps": standard_steps
                })

            except Exception as e:
                # 理论上不会再进这里
                print(f"[SKIP] Thread error: {page_url} | {e}")

    output_file = os.path.splitext(file_path)[0] + "_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")


# =======================
# 工具函数
# =======================
def is_url(string):
    return bool(re.match(r"https?://\S+", str(string)))


def is_excel_file(file_path):
    return file_path.lower().endswith((".xlsx", ".xls"))


# =======================
# 主入口
# =======================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a page URL or Excel file path.")
        sys.exit(1)

    input_arg = sys.argv[1]

    if is_url(input_arg):
        print(f"Detected page URL: {input_arg}")
        process_page(input_arg)

    elif is_excel_file(input_arg):
        print(f"Detected Excel file: {input_arg}")
        process_excel(input_arg)

    else:
        print("The provided argument is neither a valid URL nor an Excel file path.")
        sys.exit(1)
