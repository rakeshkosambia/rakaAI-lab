# rakaAI-lab

Personal AI learning lab: study material and experiments on agentic AI and agent harness engineering.

## Contents

| Folder | What it is | Origin |
|---|---|---|
| [`learn-claude-code/`](./learn-claude-code/) | 17-lesson course on building an agent harness (loop, tools, permissions, hooks, subagents, context compaction, memory, tasks, teams, MCP, workflow runtime, goal loop) | Copy of [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code), MIT. See `learn-claude-code/NOTICE.md` |
| [`learn-aws-bedrock/`](./learn-aws-bedrock/) | The same ideas rebuilt for Amazon Bedrock with `boto3`: chapters b00-b08 built and tested offline, b09-b18 planned (memory, RAG, Step Functions, AgentCore, deployment) | Original work. Chapter ideas map to the Claude Code course; see its README |

## Suggested reading order

1. Skim `learn-claude-code/README.md` for the model-versus-harness idea.
2. Work through `learn-aws-bedrock/` chapters b00 to b08 (offline first, then live on your own AWS account).
3. Return to `learn-claude-code/` lessons s09 onward for the ideas the second wave of the AWS course will cover.

## Licensing

Each folder keeps its own license terms. `learn-claude-code/` is MIT-licensed by shareAI Lab and its `LICENSE` file must stay with it. `learn-aws-bedrock/` has no license chosen yet; add one before publishing.

## Secrets

Never commit API keys, AWS credentials or `.env` files. Each project provides a `.env.example` with empty placeholders only.
