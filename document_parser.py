from selenium.webdriver.common.by import By
import json
import time

def extract_steps_from_left_pane(driver, wait_time=2):
    time.sleep(wait_time)
    lis = driver.find_elements(By.CSS_SELECTOR, ".left-pane-content.row li")
    
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

def extract_left_pane_content(driver, wait_time=10):
    time.sleep(wait_time)
    videos = driver.find_elements(By.TAG_NAME, "video")
    if videos:
        for video in videos:
            src = video.get_attribute("src")
            if not src:
                # 有些视频在 <source> 里
                source_tags = video.find_elements(By.TAG_NAME, "source")
                for source in source_tags:
                    src = source.get_attribute("src")
                    if src:
                        break
            if src:
                result.append({
                    "type": "video",
                    "src": src
                })
        return result

    # === 2️⃣ 否则提取 li 的文字和图片 ===
    lis = container.find_elements(By.TAG_NAME, "li")
    for li in lis:
        text = li.text.strip()
        try:
            img_tag = li.find_element(By.TAG_NAME, "img")
            img = img_tag.get_attribute("src")
        except:
            img = None

        result.append({
            "type": "step",
            "text": text,
            "img": img
        })

    return result