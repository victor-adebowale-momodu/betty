"""Betty command-line interface.

Orchestrates the research pipeline. All heavy logic lives in the sibling
modules; this file only wires commands together and prints concise output.

Commands:
    inspect     Inspect the raw dataset.
    process     Build the processed (primary) dataset.
    analyze     Run core market / calibration analysis and save figures/summary.
    backtest    Run the predefined strategies and print performance.
    all         Run the whole pipeline in order.
"""

from __future__ import annotations

import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Pipeline steps (thin wrappers over the real modules)
# ---------------------------------------------------------------------------

def cmd_inspect() -> None:
    from . import data
    info = data.inspect()
    print(f"Raw files ({len(info['files'])}):")
    for f in info["files"]:
        print(f"  - {f}")
    print(f"\nRows (all files): {info['n_rows']}")
    print("\nMatches by season:")
    for s, n in info["season_counts"].items():
        print(f"  {s}: {n}")
    print("\nBet365 closing-odds availability:")
    for s, has in info["has_b365_closing"].items():
        print(f"  {s}: {'yes' if has else 'no'}")
    print(f"\nMerged columns ({len(info['raw_columns'])}):")
    print("  " + ", ".join(info["raw_columns"][:20]) + " ...")


def cmd_process() -> None:
    from . import data
    df = data.process()
    print(f"Processed dataset saved to {data.PROCESSED_PATH}")
    print(f"Primary rows: {len(df)}")
    print("Season counts:")
    for s, n in df["season"].value_counts().sort_index().items():
        print(f"  {s}: {n}")


def cmd_analyze() -> None:
    from . import data, probabilities, calibration
    df = probabilities.add_probability_columns(data.load_processed())
    print("Overround summary (primary data):")
    print(df["overround"].describe().round(4).to_string())
    print(f"\nMean book percentage: {df['book_pct'].mean():.4f}")
    for o in ("H", "D", "A"):
        stats = calibration.calibration_statistics(
            calibration.probability_buckets(df, o))
        print(f"\n[{o}] calibration: {stats}")
    print("\nBest and worst overround rows by match don't drive strategy; "
          "see notebooks 02/03 for detail.")


def cmd_backtest() -> None:
    from . import data, probabilities, backtest, metrics, strategies
    df = probabilities.add_probability_columns(data.load_processed())
    print(f"{'strategy':<20}{'bets':>6}{'winrate':>9}{'avgodds':>9}"
          f"{'profit':>9}{'roi':>8}{'maxDD':>9}{'Lstrk':>6}")
    ledger_cache = {}
    for name in strategies.STRATEGIES:
        ledger = backtest.backtest_strategy(df, name)
        ledger_cache[name] = ledger
        m = metrics.summarize_ledger(ledger)
        print(f"{name:<20}{m['bets']:>6}{m['win_rate']:>9.3f}"
              f"{m['average_odds']:>9.2f}{m['net_profit']:>9.2f}"
              f"{m['roi']:>8.3f}{m['max_drawdown']:>9.2f}"
              f"{m['longest_losing_streak']:>6}")
    print("\nAll backtests are in-sample/exploratory. "
          "See notebook 05 for out-of-sample walk-forward results.")


def cmd_all() -> None:
    cmd_process()
    print()
    cmd_analyze()
    print()
    cmd_backtest()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "inspect": cmd_inspect,
    "process": cmd_process,
    "analyze": cmd_analyze,
    "backtest": cmd_backtest,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("Betty CLI\n")
        print("usage: python -m src.main <command>\n")
        print("commands:")
        for name in COMMANDS:
            print(f"  {name}")
        return 0 if argv else 1
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Available: {', '.join(COMMANDS)}")
        return 2
    COMMANDS[cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
