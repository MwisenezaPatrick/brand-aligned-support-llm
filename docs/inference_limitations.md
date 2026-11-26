# Inference limitations: Qwen3.5-0.8B Q4_K_M on the local llama.cpp server

Status: the inference mechanism works. A smoke test with thinking disabled returned a final answer with no reasoning output. The prompt harness, profile loader, and client are covered by automated tests (511 passed in 2.64s). The model's behaviour with the full system prompt has NOT been tested live. The model is not ready for a baseline evaluation.

## Resource limitation (recorded)
- Machine: 8 GB RAM, AMD Ryzen 5 5625U, CPU only.
- Settings: context 4,096 tokens, 6 threads, 127.0.0.1:8080.
- llama-server private memory: 1,295 MB. Free system RAM during that run: 0.70 GB, below the 0.8 GB gate.
- Decision: no live runs and no larger context until free RAM is restored.

## Thinking behaviour (verified)
- Without the thinking switch, the model produced reasoning output and hit the token limit with no answer (finish_reason=length).
- With chat_template_kwargs enable_thinking=false, the same request completed with an answer and no reasoning output (finish_reason=stop).
- The client sends the switch on every request by default.

## Unsupported claims (observed without the prompt)
- Three follow-up tests ran without the system prompt. They describe the bare model, not the prompt.
- Without the prompt: an invented one-year blender warranty, refund assumptions for a cracked flask, and an unsupported Saturday delivery claim.
- The prompt now prohibits these topics and routes them to a team. Whether the model obeys the prompt is unknown until a live test runs.

## Escalation and multi-topic handling
- One routed topic gets the single-team sentence. Several distinct teams get one sentence naming each team once, in team_order. Topics that share a team collapse to one team.
- The client and the builder are tested. Whether the model writes these sentences correctly is not tested.

## Safety and injury: escalation only (decision)
- No emergency sentence and no emergency trigger are implemented. Messages about fire, injury, electric shock, or gas are routed to a senior team member in one sentence.
- This is not an adequate emergency response. The assistant must not be used for safety-critical messages.

## Prompt size
- The rendered prompt was measured at 5,534 characters, below the 8,000-character cap. No unresolved placeholders remained.
- At about 2,000 tokens, prompt processing (roughly 97 tokens per second on this machine, as measured in the smoke test) would take about 20 to 25 seconds per request before the first token. Measure this in the live test.

## Known limitations deferred to a later revision
1. Brand name. "NairobiTech (demo)" is a placeholder. My search found no company by that exact name, but that does not rule one out. Before any external use, confirm the name is not a real company's, or rename it.
2. Source scan. The test named test_no_module_calls_the_raw_client_outside_the_wrapper uses a regex over four folders (training, evaluation, app, scripts). It misses aliases, getattr calls, direct httpx use, code in other folders, and notebooks. It is a tripwire, not enforcement.

## Other open items
- Unsupported business claims, wrong routing, and wrong escalation depend on the model. No deterministic output screen exists. Decision pending.
- Reasoning written as plain text without tags is not detected by the client.
- Language coverage: the synthetic data is English only.
- Evaluation: the test split is 8 synthetic conversations from 2 customers. Results describe that sample only, and they do not test the 90% or 95% targets.
- Automated tests: 511 passed in 2.64s using the project virtual environment.
