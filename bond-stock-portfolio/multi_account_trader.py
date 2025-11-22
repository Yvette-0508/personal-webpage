#!/usr/bin/env python3
"""
Utility to fan out synchronized market orders across multiple Alpaca paper accounts.

Each account's API keys and target allocations live in a YAML config (see
`portfolio_config.example.yaml`). The script:

1. Loads the config and instantiates one REST client per account.
2. Pulls up-to-date account metrics (equity or cash) to compute notionals.
3. Submits fractional-capable market orders in parallel (or dry-runs).

Example:
    python multi_account_trader.py \
        --config /absolute/path/to/portfolio_config.yaml \
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from alpaca_trade_api.rest import APIError, REST

AllocationBasis = Literal["equity", "cash"]


@dataclass
class OrderAllocation:
    """Represents a single symbol allocation for an account."""

    symbol: str
    percentage: float
    side: Literal["buy", "sell"] = "buy"
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    allocation_basis: Optional[AllocationBasis] = None
    min_notional: float = 1.0

    def normalized_basis(self, fallback: AllocationBasis) -> AllocationBasis:
        return self.allocation_basis or fallback


@dataclass
class AccountConfig:
    """Holds API credentials and per-symbol allocations for one account."""

    name: str
    key_id: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    allocation_basis: AllocationBasis = "equity"
    max_notional_per_order: Optional[float] = None
    allocations: List[OrderAllocation] = field(default_factory=list)


def load_config(path: Path) -> List[AccountConfig]:
    raw = yaml.safe_load(path.read_text())
    accounts: List[AccountConfig] = []
    for entry in raw.get("accounts", []):
        allocations = [
            OrderAllocation(
                symbol=item["symbol"],
                percentage=float(item["percentage"]),
                side=item.get("side", "buy"),
                time_in_force=item.get("time_in_force", "day"),
                allocation_basis=item.get("allocation_basis"),
                min_notional=float(item.get("min_notional", 1.0)),
            )
            for item in entry.get("allocations", [])
        ]
        accounts.append(
            AccountConfig(
                name=entry["name"],
                key_id=entry["key_id"],
                secret_key=entry["secret_key"],
                base_url=entry.get("base_url", "https://paper-api.alpaca.markets"),
                allocation_basis=entry.get("allocation_basis", "equity"),
                max_notional_per_order=entry.get("max_notional_per_order"),
                allocations=allocations,
            )
        )
    if not accounts:
        raise ValueError("No accounts found in config. Check the YAML structure.")
    return accounts


def _value_for_basis(account: Any, basis: AllocationBasis) -> float:
    if basis == "cash":
        return float(account.cash)
    return float(account.equity)


def _submit_orders_for_account(
    cfg: AccountConfig, dry_run: bool, allow_fractional: bool
) -> Dict[str, Any]:
    rest = REST(key_id=cfg.key_id, secret_key=cfg.secret_key, base_url=cfg.base_url)
    account = rest.get_account()
    account_summary = {
        "account": cfg.name,
        "submitted": [],
        "skipped": [],
    }
    for alloc in cfg.allocations:
        basis = alloc.normalized_basis(cfg.allocation_basis)
        reference_value = _value_for_basis(account, basis)
        notional = reference_value * alloc.percentage
        if cfg.max_notional_per_order:
            notional = min(notional, float(cfg.max_notional_per_order))
        if notional < alloc.min_notional:
            account_summary["skipped"].append(
                {
                    "symbol": alloc.symbol,
                    "reason": f"notional {notional:.2f} below min {alloc.min_notional}",
                }
            )
            continue
        payload = {
            "symbol": alloc.symbol,
            "side": alloc.side,
            "type": "market",
            "time_in_force": alloc.time_in_force,
        }
        if allow_fractional:
            payload["notional"] = round(notional, 2)
        else:
            latest_price = float(rest.get_latest_trade(alloc.symbol).price)
            qty = int(notional // latest_price)
            if qty == 0:
                account_summary["skipped"].append(
                    {
                        "symbol": alloc.symbol,
                        "reason": f"qty would be 0 with latest price {latest_price}",
                    }
                )
                continue
            payload["qty"] = qty
        logging.info(
            "[%s] %s %s %s %s (basis=%s, notional=%.2f)",
            cfg.name,
            "DRY-RUN" if dry_run else "LIVE",
            alloc.side.upper(),
            payload.get("qty", ""),
            alloc.symbol,
            basis,
            notional,
        )
        if dry_run:
            account_summary["submitted"].append({**payload, "dry_run": True})
            continue
        try:
            order = rest.submit_order(**payload)
            account_summary["submitted"].append({"symbol": alloc.symbol, "id": order.id})
        except APIError as exc:
            logging.exception(
                "Failed to submit order for %s on %s: %s",
                alloc.symbol,
                cfg.name,
                exc,
            )
            account_summary["skipped"].append(
                {"symbol": alloc.symbol, "reason": str(exc)}
            )
    return account_summary


async def fan_out_orders(
    accounts: List[AccountConfig], dry_run: bool, allow_fractional: bool
) -> List[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            None, _submit_orders_for_account, cfg, dry_run, allow_fractional
        )
        for cfg in accounts
    ]
    return await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute synchronized market orders across Alpaca paper accounts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Absolute path to YAML config with account allocations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log payloads without sending orders.",
    )
    parser.add_argument(
        "--disallow-fractional",
        action="store_true",
        help="Force whole-share orders (requires latest trade lookup).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config_path: Path = args.config.expanduser().resolve()
    accounts = load_config(config_path)
    logging.info("Loaded %d account definitions from %s", len(accounts), config_path)
    try:
        summaries = asyncio.run(
            fan_out_orders(
                accounts=accounts,
                dry_run=bool(args.dry_run),
                allow_fractional=not args.disallow_fractional,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Order fan-out failed: %s", exc)
        raise SystemExit(1) from exc
    for summary in summaries:
        logging.info(
            "Account %s: %d submitted, %d skipped",
            summary["account"],
            len(summary["submitted"]),
            len(summary["skipped"]),
        )


if __name__ == "__main__":
    main()

