"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"


def _get_client() -> MlflowClient:
    """Return a configured MlflowClient (reads MLFLOW_TRACKING_URI from env)."""
    return MlflowClient()


def _find_version(client: MlflowClient, name: str, config_id: str):
    """Find the registered ModelVersion whose config_id tag matches.

    Returns the ModelVersion object, or calls sys.exit(1) on zero matches.
    Prints a warning and picks the latest if multiple matches exist.
    """
    versions = client.search_model_versions(
        f"name = '{name}' AND tags.config_id = '{config_id}'"
    )
    if not versions:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    if len(versions) > 1:
        mlflow_versions = sorted([int(v.version) for v in versions])
        latest = mlflow_versions[-1]
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {mlflow_versions}); using latest ({latest})"
        )
        return next(v for v in versions if int(v.version) == latest)
    return versions[0]


def _get_current_config_id(client: MlflowClient, name: str, alias: str) -> str:
    """Return the config_id tag of whatever version the alias points at.

    Returns "" if the alias is unset.
    """
    try:
        mv = client.get_model_version_by_alias(name, alias)
        return mv.tags.get("config_id", "")
    except RestException:
        return ""


def _append_log(event: dict) -> None:
    """Append one JSON-line event to the promotion log."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _read_log() -> list[dict]:
    """Read the promotion log. Returns [] if the file doesn't exist."""
    if not LOG_FILE.exists():
        return []
    entries = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    client = _get_client()
    name = args.name

    # 1. Find the version with the given config_id
    mv = _find_version(client, name, args.config_id)

    # 2. Look up what the alias currently points at
    current_config_id = _get_current_config_id(client, name, args.alias)

    # 3. Assign the alias
    client.set_registered_model_alias(name, args.alias, mv.version)

    # 4. Append to audit log
    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current_config_id,
        "to": args.config_id,
        "op": "set",
    })

    # 5. Print summary
    from_str = current_config_id if current_config_id else "(unset)"
    print(f"{args.alias}: {from_str} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    client = _get_client()
    name = args.name

    try:
        mv = client.get_model_version_by_alias(name, args.alias)
    except RestException:
        print(f"error: alias '{args.alias}' is not set on {name}")
        sys.exit(1)

    config_id = mv.tags.get("config_id", "(unknown)")
    model = mv.tags.get("model", "(unknown)")

    # Fetch key metrics from the source eval run
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics

    accuracy = metrics.get("accuracy_overall", 0.0)
    leaked = metrics.get("verdict_rate_leaked", 0.0)
    cost = metrics.get("total_cost_usd", 0.0)

    print(f"{name} @ {args.alias}")
    print(f"  config_id: {config_id}")
    print(f"  model: {model}")
    print(f"  accuracy_overall: {accuracy:.2f}")
    print(f"  verdict_rate_leaked: {leaked:.2f}")
    print(f"  total_cost_usd: ${cost:.2f}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    client = _get_client()
    name = args.name

    try:
        rm = client.get_registered_model(name)
    except RestException:
        print("no aliases set")
        return

    aliases = rm.aliases
    if not aliases:
        print("no aliases set")
        return

    for alias_name, version_str in aliases.items():
        # Get the config_id tag from the version the alias points at
        mv = client.get_model_version(name, version_str)
        config_id = mv.tags.get("config_id", "(unknown)")
        print(f"{alias_name} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    client = _get_client()
    name = args.name

    # 1. Check the alias is currently set
    try:
        current_mv = client.get_model_version_by_alias(name, args.alias)
    except RestException:
        print("nothing to roll back")
        sys.exit(1)

    current_config_id = current_mv.tags.get("config_id", "")

    # 2. Find the most recent log entry for this alias
    log = _read_log()
    last_entry = None
    for entry in reversed(log):
        if entry.get("alias") == args.alias:
            last_entry = entry
            break

    if last_entry is None:
        print(f"no promotion history for alias {args.alias}")
        sys.exit(1)

    if last_entry["op"] == "rollback":
        print(
            f"error: {args.alias} was just rolled back; "
            f"no further history to walk back to"
        )
        sys.exit(1)

    # last_entry["op"] == "set"
    prev_config_id = last_entry.get("from", "")
    if not prev_config_id:
        print(f"{args.alias} has no previous target (first promotion ever)")
        sys.exit(1)

    # 3. Find the version for the previous config_id
    mv = _find_version(client, name, prev_config_id)

    # 4. Assign the alias back
    client.set_registered_model_alias(name, args.alias, mv.version)

    # 5. Append rollback event
    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current_config_id,
        "to": prev_config_id,
        "op": "rollback",
    })

    # 6. Print summary
    print(f"{args.alias}: {current_config_id} → {prev_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
