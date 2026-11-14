"""Exercise the real hidden Edge WebView2 window and desktop JS bridge."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import webview
    from emotionlab.bridge import DesktopAPI
    from emotionlab.common import atomic_json
    api=DesktopAPI()
    errors=[]
    window=webview.create_window("Emotion Studio verification",str(ROOT/"web/index.html"),js_api=api,
                                hidden=True,width=1320,height=900)
    api._window=window

    def check():
        try:
            deadline=time.monotonic()+30
            state=None
            while time.monotonic()<deadline:
                state=window.evaluate_js("({ready:document.getElementById('connection').classList.contains('ready'), rows:document.querySelectorAll('#dataset-rows tr').length, text:document.getElementById('toast').textContent})")
                if state and state["ready"] and state["rows"]==25:
                    break
                time.sleep(.25)
            if not state or not state["ready"] or state["rows"]!=25:
                raise RuntimeError("Bridge bootstrap failed: "+str(state))
            window.evaluate_js("document.querySelector('[data-page=dataset]').click()")
            if not window.evaluate_js("document.getElementById('dataset-page').classList.contains('active')"):
                raise RuntimeError("Dataset navigation failed")
            window.evaluate_js("document.querySelector('[data-page=analysis]').click()")
            if not window.evaluate_js("document.getElementById('analysis-page').classList.contains('active')"):
                raise RuntimeError("Analysis navigation failed")
            total=window.evaluate_js("document.getElementById('sample-count').textContent")
            assert total == f"{len(api._samples):,} 句"
            result={"ok":True,"engine":"Edge WebView2","dataset_rows":state["rows"],
                    "dataset_count":len(api._samples),"navigation":True}
            atomic_json(ROOT/"runtime/pywebview-check.json",result)
            print(json.dumps(result),flush=True)
        except BaseException as error:
            errors.append(str(error))
            print(str(error),file=sys.stderr,flush=True)
        finally:
            api._manager.close()
            window.destroy()
    webview.start(check,gui="edgechromium",debug=False)
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
