import asyncio
import argparse
import traceback

# The keyboard module is optional
try:
    import keyboard
except ImportError:
    keyboard = None

# Import core functionalities from our modules
import config
from evolution import main_async
from utils import (
    setup_keyboard_hooks,
    send_error,
    immediate_stop_event,
    graceful_stop_event,
)

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Evolutionary Strategy Optimizer")

    # --- (新增) 演化策略設定 ---
    es_group = parser.add_argument_group("演化策略 (ES) 設定")
    es_group.add_argument(
        "--selection-strategy",
        type=str,
        default=config.SELECTION_STRATEGY,
        choices=["plus", "comma"],
        help=f"選擇策略: 'plus'=(μ+λ), 'comma'=(μ,λ) (預設: {config.SELECTION_STRATEGY})",
    )
    es_group.add_argument(
        "--mutation-adaptation",
        type=str,
        default=config.MUTATION_ADAPTATION,
        choices=["adaptive", "fixed"],
        help=f"突變強度: 'adaptive'=適應性, 'fixed'=固定 (預設: {config.MUTATION_ADAPTATION})",
    )
    es_group.add_argument(
        "--diversity-control",
        action=argparse.BooleanOptionalAction,
        default=config.USE_DIVERSITY_CONTROL,
        help="啟用或禁用族群多樣性懲罰機制",
    )
    # --- Model and Feature Settings ---
    model_group = parser.add_argument_group("Model and Feature Settings")
    model_group.add_argument(
        "--add-ratios",
        action="store_true",
        help="Add side length ratio features",
    )
    model_group.add_argument(
        "--add-sincos",
        action="store_true",
        help="Add sin/cos features for angles",
    )
    model_group.add_argument(
        "--add-interactions",
        dest="add_interactions",
        action="store_true",
        default=True,
        help="Enable side-side and side-angle interactions (s*s, s*a)",
    )
    model_group.add_argument(
        "--add-aa-interact",
        dest="add_aa_interact",
        action="store_true",
        default=True,
        help="Enable angle-angle interactions (a*a)",
    )

    # --- Data and Training Settings ---
    training_group = parser.add_argument_group("Data and Training Settings")
    training_group.add_argument(
        "--scale-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale features for the length model",
    )
    training_group.add_argument(
        "--scale-angle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale features for the angle model",
    )

    # --- Functionality and Flow Control ---
    control_group = parser.add_argument_group("Functionality and Flow Control")
    control_group.add_argument(
        "--save-report",
        type=str,
        metavar="FILENAME",
        help="Save model coefficients and training errors to Excel",
    )
    control_group.add_argument(
        "--report-only",
        action="store_true",
        help="Only save the report, do not run optimization",
    )
    control_group.add_argument(
        "--no-compensation",
        action="store_true",
        help="Disable the shrinkage compensation model",
    )

    cli_args = parser.parse_args()

    # If the keyboard module is installed, set up the hotkeys
    if keyboard:
        setup_keyboard_hooks()
    else:
        print(
            "\n⚠️ 'keyboard' module not installed, hotkeys are disabled. Run 'pip install keyboard'."
        )

    try:
        asyncio.run(main_async(cli_args))
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected, preparing for immediate stop...")
        immediate_stop_event.set()
        graceful_stop_event.set()
    except Exception as e:
        subject = "A fatal error occurred in the main Evolutionary Strategy program"
        body = f"Error Type: {type(e).__name__}\nError Message: {e}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ {subject}")
        send_error(subject, body)
    finally:
        print("\nProgram has finished.")
