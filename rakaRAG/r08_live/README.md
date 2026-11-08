# r08: A live proof of concept on Amazon Bedrock

> *Spend a little, measure honestly, stop when the money says stop.*

Equivalent in the Bedrock course: **b18 Deploy and capstone**, in miniature. Everything before this chapter ran on fakes. This one is built to run once, for real, under a hard token cap, and to say plainly whether the real thing changes the story.

**Status: the code and the offline test are done; the live run is yours to do.** I cannot run it: `boto3` is not installed here and your AWS login is SSO. Until you run it, this chapter contains no live results, and nothing below claims any.

## Decisions (Rakesh, 2026-11-08)

| Decision | Choice |
|---|---|
| Provider | Amazon Bedrock |
| Models | a cheap model for extraction (and keywords), a stronger one for answers, a separate embedding model |
| Token cap | 20,000 for the whole run, counting input, output and embeddings |
| Questions | 15 paraphrase questions built from the concepts of the `sparring-partner-ooda` skill (blast radius, an assumption proving wrong, ten times the load, locked decisions, open items, a procedure executable without questions) |

## The question

Do real embeddings and a real extraction model change what r07 found with fakes, where every graph mode scored lower than plain keyword search and cost about 9 times as much?

## What runs, in this order

Every call goes through one budget guard. If a call could push the total over the cap, it is refused before it reaches AWS.

| Stage | Calls | What it does | Likely tokens |
|---|---|---|---|
| smoke | 2 | one tiny embedding and one tiny text call, to catch request-shape mistakes | about 60 |
| embed | 57 | embed the 24 documents and 33 retrieval questions | about 4,100 |
| extract | 24 | one cheap-model call per document builds the knowledge graph (whole-document chunks, at most 8 rows, at most 220 output tokens) | about 9,900 |
| keywords | 15 | one cheap-model call per new question (graph modes) | about 2,500 |
| answers | 2 | the stronger model answers one question twice: vector search against graph + vector. **We** check the citations, not the model | about 1,800 |

Likely total: **18,329 tokens, 92% of the cap.** That fits but with little slack, and these are estimates (1.3 tokens per word); the live run counts the real figures the service reports. The worst case (24,260) is above the cap, so the guard is real: the last stage is the first to be cut.

Two protections you can rely on:
1. A stage that builds one thing from many calls (the graph, the keywords) **starts only if all of it fits in the worst case**. A half-built graph would waste what was spent and give nothing usable.
2. The saved results file holds no prompts and no credentials.

## How to run it

1. **Install and sign in** (once):
   ```
   pip install -r requirements-live.txt
   ```
   Sign in to AWS the way you normally do. Confirm model access is enabled in your Region for one cheap Claude model, one stronger Claude model and one embedding model (Titan Text Embeddings or Cohere Embed).
2. **Fill in `.env`** from `.env.example`: `RAKA_PROVIDER=bedrock`, `AWS_REGION`, the three model ids, and keep `RAKA_MAX_TOTAL_TOKENS=20000`. `learn-aws-bedrock/b00_setup` lists the ids your account can call. Nothing is guessed for you, and the code refuses to start with an empty model id.
3. **See the plan, for free:**
   ```
   python r08_live/code.py --plan
   ```
4. **Smoke test, about 60 tokens:**
   ```
   python r08_live/code.py --live --smoke
   ```
   This is where a wrong model id, a missing model access or a request-shape mistake shows up, cheaply. The request shapes were written from AWS documentation and tested only against an offline fake, so a small fix here would not surprise me.
5. **The real run:**
   ```
   python r08_live/code.py --live --yes
   ```
   It saves `index/r08_results.json` (git-ignored). Read it again later with `python r08_live/code.py --report`.
6. **Send the printed report back to me** (or the contents of `index/r08_results.json`). I will score the predictions and write the live results into this README.

Without `--yes`, `--live` prints the plan and spends nothing. `--answers N` (0 to 3) changes how many questions get written answers; each costs about 900 tokens.

## Offline rehearsal

```
python r08_live/code.py --offline
```
Runs the identical code against a fake Bedrock that checks request shapes. It uses 17,627 tokens' worth of fake accounting, completes every stage, and costs nothing. Its numbers come from lexical fakes and are labelled "NOT real results". The test suite exercises caps from 300 to 20,000 and checks the cap is never exceeded.

## How the live results will be read

The eight-line list in `PREDICTIONS.md` was written before any live run (L1 to L7). The report scores each as HOLDS, WRONG or NOT CHECKED. Intervals come from a paired bootstrap over only 15 questions, so they will be wide; a gap is only believed if its interval excludes zero.

One prediction is already settled without AWS: **L2** (BM25 does worse on the new 15 than on the v2 paraphrase questions) depends only on BM25, which needs no model. Run offline it is 0.37 against 0.81 at k=3, so L2 holds. This was the first time BM25 saw the v3 questions, and it was after I wrote the predictions.

The new questions share only 29% of their words with their answers (the v2 "paraphrase" ones shared 58%), and only 1 of 15 shares 70% or more. That is the evidence that these are lexically harder, which is what real embeddings are supposed to help with.

## Deviations from LightRAG, stated so nobody overclaims

- Chunks are whole documents, to fit the cap. LightRAG chunks by tokens, with overlap.
- Entities and relations are matched by keyword scoring, not embeddings. LightRAG embeds them. Doing that here would cost thousands more tokens.
- No reranker model; the only re-ordering is by embedding similarity (`hybrid>vec`).
- 24 documents and 15 questions are small. LightRAG's own README positions it for large collections and cross-document questions, so this proof of concept may be too small for a graph to pay off. That would be a finding about scale, not about the idea.

## What you can and cannot claim from this chapter

See `../SHOWCASE.md`. In one line: until the live run is done, claim the method and the engineering, not a result.

## Exercises

1. Run `--plan` with `RAKA_MAX_TOTAL_TOKENS=12000` and see which stages would be cut.
2. Change `ANSWER_SAMPLE` to a question of your own and compare the two answers.
3. After the live run, open `index/r08_results.json` and find the question where the graph helped most and the one where it hurt most. What was different about them?
