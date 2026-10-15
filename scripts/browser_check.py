"""Optional browser smoke checks and genuine UI screenshots.

Extra tooling (not needed to run the project):
  python -m pip install playwright
  python -m playwright install chromium
Start python app.py in another terminal, then:
  python scripts/browser_check.py --url http://127.0.0.1:5000
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import path, ensure_dir, save_json


def main():
    from playwright.sync_api import sync_playwright
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:5000")
    args = p.parse_args()
    images = ensure_dir(path("images", "screenshots"))
    errors, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1512, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#kwh').textContent !== '—'")
        checks.append({"check": "Overview renders with model forecast", "passed": page.locator('#overview-chart svg').count()==1})
        page.screenshot(path=str(images / "01_overview_synthetic.png"), full_page=True)
        page.click('[data-page="forecast"]')
        page.click('#page-forecast [data-hours="48"]')
        page.wait_for_function("document.querySelector('#forecast-table').children.length === 48")
        checks.append({"check": "48-hour explorer and table update", "passed": True})
        page.click('[data-page="sandbox"]')
        before_saving = page.locator('#scenario-saving').inner_text()
        page.locator('#flex').evaluate("el => { el.value = '20'; }")
        page.locator('#flex').dispatch_event('input')
        page.click('#apply-scenario')
        page.wait_for_function("document.querySelector('#apply-scenario').disabled === false")
        checks.append({"check": "Scenario recalculates after slider change", "passed": 'Energy conserved' in page.locator('#energy-conserved').inner_text() and page.locator('#scenario-saving').inner_text() != before_saving and page.evaluate('state.scenario.result.settings.flexible_fraction') == 0.20})
        page.screenshot(path=str(images / "02_decision_sandbox.png"), full_page=True)
        page.click('[data-page="evidence"]')
        checks.append({"check": "Six candidate rows visible", "passed": page.locator('#comparison-table tr').count()==6})
        page.screenshot(path=str(images / "03_model_evidence.png"), full_page=True)
        page.select_option('#dataset','uci')
        page.wait_for_function("document.querySelector('#source-pill').textContent === 'PUBLIC MEASURED DATA'")
        page.wait_for_function("document.querySelector('#workspace').classList.contains('loading') === false")
        checks.append({"check": "Switch to real UCI benchmark", "passed": 'Random forest' in page.locator('#brief-model').inner_text()})
        page.click('[data-page="overview"]')
        page.click('#page-overview [data-hours="24"]')
        page.wait_for_function("document.querySelector('#horizon-label').textContent === 'Next 24 hours'")
        page.screenshot(path=str(images / "04_overview_uci.png"), full_page=True)
        with page.expect_download() as download:
            page.locator('.heading-row .artifact').click()
        checks.append({"check": "Forecast CSV downloads", "passed": download.value.suggested_filename.endswith('forecast_168h.csv')})
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(images / "05_mobile.png"),full_page=True)
        checks.append({"check": "Mobile layout has no horizontal overflow", "passed": page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')})
        snapshot = path('outputs','dashboard.html')
        if snapshot.exists():
            offline = browser.new_page(viewport={"width":1512,"height":1080})
            offline.on('pageerror',lambda e:errors.append(str(e)))
            offline.goto(snapshot.as_uri(),wait_until='load')
            offline.wait_for_function("document.querySelector('#kwh').textContent !== '—'")
            checks.append({"check":"Standalone HTML works without server/network", "passed": 'offline ready' in offline.locator('#engine-status').inner_text()})
            offline.click('[data-page="sandbox"]')
            offline.locator('#flex').evaluate("el => { el.value = '25'; el.dispatchEvent(new Event('input')); }")
            offline.click('#apply-scenario')
            offline.wait_for_function('state.scenario.result.settings.flexible_fraction === 0.25')
            checks.append({"check":"Offline scenario remains interactive","passed": 'Energy conserved' in offline.locator('#energy-conserved').inner_text()})
        browser.close()
    checks.append({"check":"No browser JavaScript exceptions","passed":not errors})
    report={"passed":all(c['passed'] for c in checks),"checks":checks,"browser_errors":errors,
            "note":"Genuine browser smoke checks; screenshots are captures of this app, not generated mockups."}
    save_json(report,path('outputs','browser-checks.json'))
    for c in checks:
        print(('PASS' if c['passed'] else 'FAIL')+' | '+c['check'])
    if errors: print(errors)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
