# learn-aws-bedrock

Learn how to build an **agent harness** on **Amazon Bedrock**, one mechanism per chapter, with code you can read, run and break.

This course follows the structure of [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) (MIT) so the two can be read side by side. The text and code here are original; the chapter ideas (agent loop, tools, permissions, hooks, and so on) are the shared, general ones. Each chapter lists which lesson it corresponds to.

## The mental model

The model is the driver; the harness is the vehicle. The Claude Code course builds the whole vehicle from parts. On AWS you build the same vehicle, but several parts are **services you rent**: permits (IAM), a logbook (CloudWatch, audit), a garage (AgentCore). So each chapter shows the raw `boto3` way first, then names the AWS-managed equivalent.

## Chapters

Built now (first wave, tested offline):

| # | Chapter | Claude Code course | What you learn |
|---|---|---|---|
| b00 | [Setup and access check](b00_setup/) | (AWS only) | identity, Region, model access, one `converse()` call |
| b01 | [The Converse loop](b01_converse_loop/) | s01 | the loop, `stopReason`, `toolUse` / `toolResult` |
| b02 | [Tool use](b02_tool_use/) | s02 | a registry, error results, a file sandbox |
| b03 | [Permissions](b03_permission/) | s03 | code-level policy plus IAM least privilege |
| b04 | [Hooks](b04_hooks/) | s04 | audit log, token budget, secret masking, Guardrails |
| b05 | [Plan and todo](b05_todo_plan/) | s05 | a visible plan and a gentle nag |
| b06 | [Subagents](b06_subagent/) | s06 | clean-context delegation |
| b07 | [Skill loading](b07_skill_loading/) | s07 | on-demand knowledge from disk or S3 |
| b08 | [Context compaction](b08_context_compact/) | s08 | shrinking history without breaking tool pairs |

Planned (second wave, built after you have run the first wave against a real account):

| # | Chapter | Equivalent | AWS pieces |
|---|---|---|---|
| b09 | Memory | s09 | DIY (DynamoDB/S3) vs AgentCore Memory |
| b10 | Knowledge and RAG | (AWS only) | Bedrock Knowledge Bases, embeddings |
| b11 | Task state | s10 | persisted task graph |
| b12 | Async work | s11 | queues and functions |
| b13 | Scheduling | s12 | EventBridge Scheduler |
| b14 | Multi-agent | s13 | Strands patterns, A2A |
| b15 | Tool gateway and MCP | s14 | AgentCore Gateway vs plain MCP |
| b16 | Workflow runtime | s16 | Step Functions with an agent inside the states |
| b17 | Evaluator loop | s17 | independent judge, AgentCore Evaluations |
| b18 | Deploy and capstone | s15 | AgentCore Runtime, Identity, Observability |

Bedrock Agents (Classic) is **not** a basis for this course: AWS documents it as closed to new customers and in maintenance mode, and points new users to AgentCore.

## Quick start

Offline needs only Python 3.10+ and nothing else installed except `pip install -r requirements.txt` (and `python-dotenv` for `.env`, already listed).

```
pip install -r requirements.txt
python b01_converse_loop/code.py --offline
```

Windows wrappers (optional; set `PYTHON_EXE` to a specific Python if you want):

```
run.bat b01_converse_loop --offline
test.bat
```

Run the chapters in order. Each folder has a `README.md` (the lesson) and a `code.py` (a complete, runnable program).

## Going live

1. Copy `.env.example` to `.env`. **Never commit `.env`.**
2. Sign in to AWS the way you normally do, then set `AWS_PROFILE` and `AWS_REGION`.
3. Run `b00_setup/code.py`. It lists what your account can call and tests one request.
4. Put one model or inference-profile ID in `BEDROCK_MODEL_ID`.
5. Run any chapter without `--offline`.

### Cost and safety

- Live runs spend tokens on **your** account. Start with a small, inexpensive model.
- `LAB_MAX_TOTAL_TOKENS` (default 50,000) stops a runaway loop; b04 enforces it as a hook.
- Use an IAM role with no admin rights. See `iam/` and chapter b03.
- Offline mode costs nothing and needs no account.

## Offline mode

`--offline` replaces Bedrock with `FakeBedrockRuntime` (`lab_common.py`). It replays a scripted conversation and **rejects malformed requests** the way Converse would: wrong role order, a `toolResult` that does not match a `toolUse`, missing `toolConfig`. That makes the tests meaningful, but it is a teaching aid, not a copy of the service.

## Repository layout

```
lab_common.py      shared helpers, the offline fake, the request validator
lab_tools.py       tool registry and sandboxed file tools (the core of b02)
bNN_*/             one chapter: README.md + code.py
skills/            sample skills used by b07
iam/               IAM policy templates (placeholders only)
workspace/         sample files the file tools may read
tests/             offline tests (pytest)
```

## What is verified, and what is not

- **Verified here:** every chapter runs offline end to end; the offline test suite passes; no secrets or account identifiers are in the repo.
- **Not yet verified:** live runs against a real AWS account. The Converse request and response shapes follow the public AWS documentation and examples, but this code has only been exercised against the stub. Chapter READMEs end with a **Verify** list of the things to confirm on your account (model availability, IAM actions, guardrail API details).

## License

No license has been chosen for the original code in this folder. Add a `LICENSE` file of your choice before publishing. Do not copy text or code from other projects into this folder without keeping their license.
