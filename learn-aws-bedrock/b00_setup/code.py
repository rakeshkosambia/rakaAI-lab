"""b00: Setup and access check.

Motto: find out what works BEFORE you build on it.

Live mode checks, in order:
  1. who you are in AWS (sts get-caller-identity)
  2. which Anthropic models and inference profiles your account can see
  3. whether one tiny converse() call succeeds with BEDROCK_MODEL_ID

Offline mode skips AWS and only shows the shape of a converse() round trip.

Run:  python b00_setup/code.py --offline
      python b00_setup/code.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab_common as lc  # noqa: E402

DEFAULT_PROMPT = "Say hello in exactly five words."


def mask_account(account_id: str) -> str:
    """Show only the last 4 digits so screenshots and logs do not leak the full ID."""
    return "********" + account_id[-4:] if len(account_id) >= 4 else "****"


def check_identity(session) -> dict:
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    return {"account": mask_account(identity["Account"]), "arn": identity["Arn"]}


def list_anthropic_models(session) -> list[dict]:
    """Foundation models from Anthropic that this Region offers."""
    bedrock = session.client("bedrock")
    response = bedrock.list_foundation_models(byProvider="Anthropic")
    rows = []
    for summary in response.get("modelSummaries", []):
        rows.append(
            {
                "modelId": summary.get("modelId"),
                "name": summary.get("modelName"),
                "inference": ",".join(summary.get("inferenceTypesSupported", [])),
                "status": (summary.get("modelLifecycle") or {}).get("status", ""),
            }
        )
    return sorted(rows, key=lambda r: r["modelId"] or "")


def list_anthropic_profiles(session) -> list[dict]:
    """Inference profiles (cross-Region routes) whose ID or name mentions Anthropic/Claude."""
    bedrock = session.client("bedrock")
    rows: list[dict] = []
    token = None
    while True:
        kwargs = {"maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        response = bedrock.list_inference_profiles(**kwargs)
        for profile in response.get("inferenceProfileSummaries", []):
            text = (profile.get("inferenceProfileId", "") + profile.get("inferenceProfileName", "")).lower()
            if "anthropic" in text or "claude" in text:
                rows.append(
                    {
                        "profileId": profile.get("inferenceProfileId"),
                        "name": profile.get("inferenceProfileName"),
                        "status": profile.get("status"),
                    }
                )
        token = response.get("nextToken")
        if not token:
            break
    return sorted(rows, key=lambda r: r["profileId"] or "")


def smoke_converse(client, model_id: str, prompt: str = DEFAULT_PROMPT) -> dict:
    """One minimal converse() call. This is the whole API in 6 lines."""
    response = client.converse(
        modelId=model_id,
        messages=[lc.user_text(prompt)],
        inferenceConfig={"maxTokens": 60},
    )
    message = response["output"]["message"]
    return {
        "text": lc.message_text(message),
        "stopReason": response["stopReason"],
        "usage": response.get("usage", {}),
    }


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title} ({len(rows)})")
    if not rows:
        print("  (none found)")
    for row in rows:
        print("  " + "  |  ".join(str(v) for v in row.values()))


def main(argv: list[str] | None = None) -> int:
    lc.load_env()
    parser = lc.common_parser(__doc__.splitlines()[0], DEFAULT_PROMPT)
    args = parser.parse_args(argv)

    if args.offline:
        print("OFFLINE: skipping identity and model listing (they need a real AWS account).")
        client = lc.make_runtime_client(True, [lc.text_response("Hello there, friendly offline world.")])
        result = smoke_converse(client, lc.get_model_id(True), args.prompt)
        print("converse() round trip (scripted):", result)
        print("\nRun without --offline once your .env is filled in.")
        return 0

    exit_code = 0
    try:
        session = lc.make_session()
        identity = check_identity(session)
        print(f"Signed in as: {identity['arn']}")
        print(f"Account:      {identity['account']}")
        print(f"Region:       {session.region_name}")
    except Exception as exc:  # noqa: BLE001
        print("Identity check FAILED:", lc.explain_aws_error(exc))
        return 1

    try:
        print_table("Anthropic foundation models in this Region", list_anthropic_models(session))
    except Exception as exc:  # noqa: BLE001
        print("\nModel listing FAILED:", lc.explain_aws_error(exc))
        exit_code = 1
    try:
        print_table("Anthropic inference profiles", list_anthropic_profiles(session))
    except Exception as exc:  # noqa: BLE001
        print("\nProfile listing FAILED:", lc.explain_aws_error(exc))
        exit_code = 1

    model_id = os.getenv("BEDROCK_MODEL_ID", "").strip()
    if not model_id:
        print("\nBEDROCK_MODEL_ID is empty. Pick an ID from the lists above, put it in .env, and run this again.")
        return exit_code or 2
    print(f"\nTrying one converse() call with {model_id} ...")
    try:
        client = lc.make_runtime_client(False)
        result = smoke_converse(client, model_id, args.prompt)
        print("Reply:      ", result["text"])
        print("Stop reason:", result["stopReason"])
        print("Tokens:     ", result["usage"])
        print("\nAll good. Continue with b01_converse_loop.")
    except Exception as exc:  # noqa: BLE001
        print("converse() FAILED:", lc.explain_aws_error(exc))
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
