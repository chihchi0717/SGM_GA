import asyncio
import argparse
import traceback

# 鍵盤模組為選用
try:
    import keyboard
except ImportError:
    keyboard = None

# 從我們的模組中匯入核心功能
from evolution import main_async
from utils import (
    setup_keyboard_hooks,
    send_error,
    immediate_stop_event,
    graceful_stop_event,
)

if __name__ == "__main__":
    # 設定命令列參數解析器
    parser = argparse.ArgumentParser(description="演化策略優化器")
    parser.add_argument("--add-ratios", action="store_true", help="加入邊長比例特徵")
    parser.add_argument(
        "--add-sincos", action="store_true", help="加入角度的 sin/cos 特徵"
    )
    parser.add_argument(
        "--no-interactions",
        dest="add_interactions",
        action="store_false",
        help="禁用邊長與角度交互作用 (s*s, s*a)",
    )
    parser.add_argument(
        "--no-aa-interact",
        dest="add_aa_interact",
        action="store_false",
        help="禁用角度間交互作用 (a*a)",
    )
    parser.add_argument(
        "--save-report", type=str, metavar="FILENAME", help="將模型係數儲存至 Excel"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="僅儲存報告，不執行優化"
    )
    parser.add_argument(
        "--no-compensation", action="store_true", help="禁用收縮補償模型"
    )
    parser.set_defaults(add_interactions=True, add_aa_interact=True)
    cli_args = parser.parse_args()

    # 如果安裝了 keyboard 模組，就設定快速鍵
    if keyboard:
        setup_keyboard_hooks()
    else:
        print(
            "\n⚠️  未安裝 'keyboard' 模組，無法使用快速鍵停止。請執行 'pip install keyboard'。"
        )

    try:
        # 執行主非同步迴圈
        asyncio.run(main_async(cli_args))
    except KeyboardInterrupt:
        print("\n🛑 偵測到 Ctrl+C，正在準備立即停止...")
        immediate_stop_event.set()
        graceful_stop_event.set()
    except Exception as e:
        subject = "演化策略主程式發生致命錯誤"
        body = f"錯誤類型: {type(e).__name__}\n錯誤訊息: {e}\n\n追蹤訊息:\n{traceback.format_exc()}"
        print(f"❌ {subject}")
        send_error(subject, body)
    finally:
        print("\n程式已結束。")
