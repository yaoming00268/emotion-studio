"""Headless layout and UI checks using the actual desktop bridge, isolated browser."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    args=parser.parse_args()
    from playwright.sync_api import sync_playwright
    from emotionlab.bridge import DesktopAPI
    api=DesktopAPI()
    out=ROOT/"runtime/previews"
    out.mkdir(parents=True,exist_ok=True)
    errors=[]
    report=None
    try:
        report=verify(api,args,out,errors)
    finally:
        api._manager.close()
    if errors:
        raise RuntimeError("; ".join(errors))
    from emotionlab.common import atomic_json
    atomic_json(ROOT/"runtime/ui-check.json",report)
    print(json.dumps(report,ensure_ascii=False))


def verify(api,args,out,errors):
    from playwright.sync_api import sync_playwright
    bootstrap=api.bootstrap()["data"]
    initial_count=bootstrap["dataset"]["count"]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(channel="msedge",headless=True)
        page=browser.new_page(viewport={"width":1440,"height":1080},device_scale_factor=1)
        page.on("pageerror",lambda error: errors.append(str(error)))
        page.expose_function("desktopCall",lambda name,args: getattr(api,name)(*args))
        page.add_init_script("window.pywebview={api:new Proxy({}, {get:(_,name)=>(...args)=>window.desktopCall(name,args)})}; document.addEventListener('DOMContentLoaded',()=>window.dispatchEvent(new Event('pywebviewready')));")
        page.goto((ROOT/"web/index.html").as_uri())
        page.wait_for_function("document.querySelectorAll('#dataset-rows tr').length === 25")
        page.screenshot(path=str(out/"training.png"),full_page=True)
        page.locator("[data-page=dataset]").click()
        assert page.locator("#dataset-rows tr").count()==25
        if bootstrap["expansion"]["ready"]:
            page.locator("#use-expanded").click()
            page.wait_for_function("document.getElementById('toast').textContent.includes('启用均衡抽样') && !document.getElementById('use-expanded').disabled")
            initial_count=bootstrap["expansion"]["train_count"]
            page.wait_for_function("expected => document.getElementById('sample-count').textContent === expected",arg=f"{initial_count:,} 句")
            assert page.locator("#balance_classes").is_checked()
            assert page.locator("#dataset-rows .sample-provenance").count()==25
            assert page.locator("#corpus-quality").is_visible()
            assert page.locator("#corpus-sources .corpus-bar").count()>=2
            # Filtering must edit the underlying index, rather than the row's
            # visible position in a filtered page.
            page.select_option("#filter-label", "悲伤")
            page.wait_for_function("Array.from(document.querySelectorAll('#dataset-rows select')).every(x => x.value === '悲伤')")
            page.locator("#dataset-rows summary").first.click()
            assert page.locator("#dataset-rows .provenance-details").first.is_visible()
            page.screenshot(path=str(out/"acg-filtered-review.png"),full_page=True)
            page.select_option("#filter-label", "")
            page.wait_for_function("expected => document.getElementById('page-number').textContent === expected && document.querySelector('#dataset-rows tr td').textContent === '0001'",arg=f"1 / {(initial_count+24)//25}")
            page.screenshot(path=str(out/"expanded-annotation.png"),full_page=True)
        first=page.locator("#dataset-rows tr").first
        first.locator("td").nth(1).fill("这是一句等待保存的分页标注检查。")
        first.locator("select").select_option("喜悦")
        page.locator("#next-page").click()
        page.wait_for_function("document.getElementById('page-number').textContent.startsWith('2 /')")
        edited=api.dataset_page(0,1)["data"]["rows"][0]
        assert edited["text"]=="这是一句等待保存的分页标注检查。"
        assert edited["label"]=="喜悦"
        page.locator("#new-samples").fill("新添加的句子需要逐一检查。今天我很开心！")
        page.locator("#add-samples").click()
        page.wait_for_function("expected => document.getElementById('sample-count').textContent === expected", arg=f"{initial_count+2:,} 句")
        assert page.locator("#dataset-rows select").last.input_value()==""
        page.screenshot(path=str(out/"annotation.png"),full_page=True)
        page.locator("[data-page=analysis]").click()
        if args.checkpoint:
            page.select_option("#checkpoint",str(Path(args.checkpoint).resolve()))
            page.locator("#view-run-metrics").click()
            page.wait_for_function("document.getElementById('accuracy-value').textContent !== '—'")
            page.wait_for_timeout(1500)
            assert page.locator("#accuracy-value").inner_text()!="—"
            page.screenshot(path=str(out/"training.png"),full_page=True)
            page.locator("[data-page=analysis]").click()
            page.locator("#load-analysis").click()
            page.wait_for_function("document.getElementById('analysis-status').textContent.includes('已连接')",timeout=30000)
            page.locator("#analysis-text").fill("今天收到朋友的祝福，我真的很开心！\n有人无缘无故冤枉我，气死我了！\n会议明天下午两点开始。")
            page.locator("#predict").click()
            page.wait_for_function("document.querySelectorAll('.prediction').length === 3",timeout=180000)
        page.screenshot(path=str(out/"analysis.png"),full_page=True)
        for width in (1060,1320):
            page.set_viewport_size({"width":width,"height":900})
            for nav in ("training","dataset","analysis"):
                page.locator(f"[data-page={nav}]").click()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),f"Overflow: {nav} @ {width}"
        browser.close()
    return {"ok":True,"screenshots":[str(path) for path in out.glob("*.png")],
            "dataset_count":initial_count,"expanded_selection":bootstrap["expansion"]["ready"],
            "annotation_edit_pagination":True,"actual_inference":bool(args.checkpoint)}


if __name__=="__main__":
    main()
