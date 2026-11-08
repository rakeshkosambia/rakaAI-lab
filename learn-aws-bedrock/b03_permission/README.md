# b03: Permissions

> *Set the boundaries first, then grant freedom.*

Equivalent in the Claude Code course: **s03 Permission**. On AWS there is a second, stronger layer that the original course does not have: IAM.

## The problem

"The model asked for it" must never mean "it happened". Models can be wrong, tricked by text inside a file they read, or simply over-eager.

## The idea: two locks on one door

- **Lock 1, in your code**: `PermissionPolicy` decides `allow`, `ask` or `deny` before any tool runs. Fast, friendly messages, works offline.
- **Lock 2, in AWS**: an IAM policy limits what the credentials can do even if lock 1 has a bug. The model cannot talk its way past IAM.

You want both. Code-level rules are about *intent* ("do not read `.env`"); IAM is about *capability* ("this role cannot touch any other bucket").

## Lock 1: the policy

| Mode | Reads | Writes |
|---|---|---|
| `readonly` | allowed | denied |
| `default` | allowed | **ask a human** (no console, no approval) |
| `auto` | allowed | allowed inside the sandbox |

Always denied, in every mode: files matching `.env`, `*.pem`, `*.key`, `*credentials*`, `*secret*`, `id_rsa*`. **Deny beats allow.**

A denied call returns `PERMISSION DENIED: <reason>` as an error result. The model sees why and can pick another route. Tools declare `risk="read"` or `risk="write"` at registration; the policy keys off that.

## Lock 2: IAM templates (`iam/`)

- `bedrock-invoke-minimal.json`: lets the role invoke only the models you name. Replace `REGION`, `MODEL_ID`, `ACCOUNT_ID`, `INFERENCE_PROFILE_ID`.
- `s3-read-prefix.json`: read-only access to one S3 prefix (used by b07 and, later, knowledge bases).

Practical habits:

- Run lessons with a role that has **no** admin rights and no write access to anything you care about.
- Start from these templates, then test them with the IAM policy simulator in the AWS console before relying on them.
- Never put an access key in code, `.env.example`, or a prompt.

## Run it

```
python b03_permission/code.py --offline
python b03_permission/code.py --mode readonly --prompt "Save a note called todo.txt"
run.bat b03_permission --mode default
```

## Try this

1. Run `--mode readonly` live. The write is denied; read the model's explanation.
2. Add `*.sql` to `DENY_PATTERNS` and ask it to read a `.sql` file.
3. Change `write_text_file` to `risk="read"` and notice what silently changes. This is why risk labels deserve code review.

## Pitfalls

- **Matching on names only.** The policy checks file *names*; the sandbox (b02) checks *locations*. Keep both.
- **`ask` without a human.** Unattended runs must default to deny. Here a non-interactive session answers no.
- **Over-broad IAM.** `bedrock:*` on `*` is convenient and wrong.

## Verify on your own account

- Which IAM actions Converse needs: my understanding is that `converse()` is authorised through `bedrock:InvokeModel` on the model, but confirm in the current Bedrock IAM documentation.
- Inference profiles can route to other Regions. The policy for a profile may also need permission on the underlying foundation-model ARNs in the destination Regions. Test with the policy simulator and the b00 smoke call.
- Your organisation may add service control policies that override these templates.
