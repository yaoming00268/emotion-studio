"""Native Edge WebView2 shell. Browser assets are entirely local."""
from .common import PROJECT_ROOT


def main(debug=False):
    import webview
    from .bridge import DesktopAPI
    api = DesktopAPI()
    window = webview.create_window("Emotion Studio · BERT 情感工作台", str(PROJECT_ROOT / "web/index.html"),
        js_api=api, width=1320, height=900, min_size=(1060, 720), background_color="#f5f5f2", text_select=True)
    api._window = window
    window.events.closed += api._manager.close
    webview.start(gui="edgechromium", debug=debug)
