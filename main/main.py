import asyncio
import argparse
import traceback

# 鍵盤模組為選用
try:
    import keyboard
except ImportError:
    keyboard = None

# 從我們的模組中匯入核心功能
import config
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

    # --- 模型與特徵設定 ---
    model_group = parser.add_argument_group("模型與特徵設定")
    model_group.add_argument(
        "--model-type",
        type=str,
        default=config.MODEL_TYPE,
        choices=["Huber", "RF", "OLS"],
        help=f"選擇要使用的模型類型 (預設: {config.MODEL_TYPE})",
    )
    model_group.add_argument(
        "--add-ratios",
        action=argparse.BooleanOptionalAction,
        default=config.ADD_RATIOS,
        help="加入邊長比例特徵",
    )
    model_group.add_argument(
        "--add-sincos",
        action=argparse.BooleanOptionalAction,
        default=config.ADD_SINCOS,
        help="加入角度的 sin/cos 特徵",
    )
    model_group.add_argument(
        "--add-interactions",
        action=argparse.BooleanOptionalAction,
        default=config.ADD_INTERACTIONS,
        help="啟用 s*s, s*a 交互作用",
    )
    model_group.add_argument(
        "--add-aa-interact",
        action=argparse.BooleanOptionalAction,
        default=config.ADD_AA_INTERACT,
        help="啟用 a*a 交互作用",
    )

    # --- 資料與訓練設定 ---
    training_group = parser.add_argument_group("資料與訓練設定")
    training_group.add_argument(
        "--average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否對相同結構的樣本進行平均",
    )
    training_group.add_argument(
        "--scale-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否對長度模型的特徵進行縮放",
    )
    training_group.add_argument(
        "--scale-angle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否對角度模型的特徵進行縮放",
    )

    # --- 功能與流程控制 ---
    control_group = parser.add_argument_group("功能與流程控制")
    control_group.add_argument(
        "--save-report",
        type=str,
        metavar="FILENAME",
        help="將模型係數與訓練誤差儲存至 Excel",
    )
    control_group.add_argument(
        "--save-plots",
        action="store_true",
        help="在儲存報告時，同時產生並儲存誤差分析圖表",
    )
    control_group.add_argument(
        "--report-only", action="store_true", help="僅儲存報告，不執行優化"
    )
    control_group.add_argument(
        "--no-compensation", action="store_true", help="禁用收縮補償模型"
    )

    cli_args = parser.parse_args()

    # 如果安裝了 keyboard 模組，就設定快速鍵
    if keyboard:
        setup_keyboard_hooks()
    else:
        print(
            "\n⚠️  未安裝 'keyboard' 模組，無法使用快速鍵停止。請執行 'pip install keyboard'。"
        )

    try:
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
