import sys
from pathlib import Path
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

index_path = Path(__file__).resolve().parents[1] / "docs" / "index.html"
print(f"Testing URL: {index_path.as_uri()}")

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

driver = webdriver.Chrome(options=options)
driver.get(index_path.as_uri())
time.sleep(1)

def check_logs(step):
    logs = driver.get_log("browser")
    errs = [l for l in logs if l["level"] in ["SEVERE", "ERROR"]]
    if errs:
        print(f"FAILED at {step}: {errs}")
        return False
    else:
        print(f"PASSED: {step}")
        return True

passed = True
passed &= check_logs("Initial Load")

# 1. Switch tabs
for tab in ["forecast", "sandbox", "evidence", "architecture", "gallery", "runbook", "overview"]:
    driver.find_element("css selector", f'.nav-item[data-page="{tab}"]').click()
    passed &= check_logs(f"Tab: {tab}")

# 2. Switch horizon
for h in ["48", "168", "24"]:
    driver.find_element("css selector", f'#tab-overview button[data-hours="{h}"]').click()
    time.sleep(0.4)
    passed &= check_logs(f"Horizon {h}h")

# 3. Switch dataset to UCI
select = driver.find_element("id", "dataset-dropdown")
for opt in select.find_elements("tag name", "option"):
    if opt.get_attribute("value") == "uci":
        opt.click()
        break
time.sleep(0.8)
passed &= check_logs("Switch to UCI")
print("UCI KPI kWh:", driver.find_element("id", "kpi-kwh").text)
print("UCI RMSE:", driver.find_element("id", "kpi-rmse").text)

# 4. In UCI, test tabs
for tab in ["forecast", "sandbox", "evidence"]:
    driver.find_element("css selector", f'.nav-item[data-page="{tab}"]').click()
    passed &= check_logs(f"UCI Tab: {tab}")

# 5. In Sandbox, test slider and run scenario
driver.find_element("css selector", '.nav-item[data-page="sandbox"]').click()
time.sleep(0.4)
slider = driver.find_element("id", "flex-slider")
driver.execute_script("arguments[0].value = 25; arguments[0].dispatchEvent(new Event('input'));", slider)
time.sleep(0.4)
passed &= check_logs("Slider input")

run_btn = driver.find_element("id", "btn-run-scenario")
run_btn.click()
time.sleep(0.4)
passed &= check_logs("Run scenario button")

# 6. Switch back to synthetic
for opt in select.find_elements("tag name", "option"):
    if opt.get_attribute("value") == "synthetic":
        opt.click()
        break
time.sleep(0.8)
passed &= check_logs("Switch back to Synthetic")

driver.quit()
sys.exit(0 if passed else 1)
