# b00: Setup and access check

> *Find out what works before you build on it.*

Equivalent in the Claude Code course: none. That course calls one API with one key. On AWS you also need a signed-in identity, a Region, model access, and the right model ID. Most "it does not work" moments in later chapters are really setup problems, so this chapter finds them first.

## What it checks (live mode)

1. **Who you are**: `sts get_caller_identity`. Prints your ARN and only the last 4 digits of the account ID.
2. **What you can see**: Anthropic foundation models in your Region, and inference profiles that mention Claude/Anthropic.
3. **Whether a call works**: one tiny `converse()` call using `BEDROCK_MODEL_ID`.

The whole Converse API, in the smoke test:

```python
response = client.converse(
    modelId=model_id,
    messages=[{"role": "user", "content": [{"text": "Say hello in exactly five words."}]}],
    inferenceConfig={"maxTokens": 60},
)
text = response["output"]["message"]["content"][0]["text"]
```

## Run it

```
python b00_setup/code.py --offline      # no AWS needed: shows the shape of a round trip
run.bat b00_setup                       # Windows wrapper, live mode
python b00_setup/code.py                # live mode
```

Before live mode: copy `.env.example` to `.env`, set `AWS_PROFILE` and `AWS_REGION`, and sign in to AWS the way you normally do. **This repo never handles logins or stores credentials.**

## Reading the output

- **Foundation models** list: model IDs your Region offers. Being listed does not mean you may call it.
- **Inference profiles** list: an inference profile is a named route that can spread requests across Regions. Some models must be called with a profile ID instead of the plain model ID. If a plain model ID gives a `ValidationException`, try the matching profile ID.
- Put exactly one ID in `BEDROCK_MODEL_ID` and run again.

## When it fails

| You see | Likely cause | Fix |
|---|---|---|
| "credentials or SSO session expired" | Login timed out | Sign in again, retry |
| "credentials were not found" | Wrong or empty `AWS_PROFILE` | Check `.env` and your AWS config |
| "No AWS region is set" | `AWS_REGION` empty | Set it in `.env` |
| "Access denied" on listing | Role lacks `bedrock:ListFoundationModels` / `ListInferenceProfiles` | Ask for those read actions, or skip listing and set the ID by hand |
| "Access denied" on converse | Model access not enabled in this Region, or IAM does not allow invoking it | Enable model access in the Bedrock console; compare with `iam/bedrock-invoke-minimal.json` |
| "Bedrock rejected the request" | Wrong ID, needs a profile ID, or model not in this Region | Use an ID from the lists above |

## Verify on your own account

These facts depend on your account and change over time, so the chapter checks them live instead of assuming:

- Which Claude models appear in **your** Region, and whether they need an inference profile.
- Whether model access is enabled for the model you picked.
- The exact permission names your organisation requires.
