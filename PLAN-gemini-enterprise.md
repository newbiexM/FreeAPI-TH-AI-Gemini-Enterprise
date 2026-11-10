# TH-AI Passport Gemini Enterprise bridge — implementation plan

> Plan only. No changes to code, login, credentials, runtime, or services are authorized by this document.

## Goal
Extend the existing Hermes AiPASS bridge to use the Gemini Enterprise access reached through `https://de.aipass.net/ge/open`, using TH-AI Passport entitlement, running independently of the Mac inside the Hermes container, and supporting Hermes conversation and tool use when proven feasible.

## Scope and constraints
- Project: `hermes-aipass` workspace. Current inspected commit: `2111987`; recheck status before editing; untracked `skills/` must be preserved.
- Preserve existing AiPASS provider, account, functionality, state, and default model. New backend must be opt-in. Do not publish bridge or bind beyond loopback.
- User requires TH-AI Passport's own entitlement and container runtime without ongoing Mac dependency. Separate paid Google Cloud entitlement is not a substitute. The user has not answered whether login may initially require the Mac, nor whether an unofficial interface is acceptable.
- The established browser security boundary prohibits copying cookies, browser storage, profiles, or tokens from the Mac to Linux. Do not ask user to paste credentials into chat. If no supported authorization mechanism meets both constraints, stop: seek the user's decision on an approved design; do not silently weaken the boundary.
- `/ge/open` is a login-gated entry route, NOT a verified model endpoint. The redirect, target tenant, model interface, auth, quota, and rights have not been inspected. Never infer that AiPASS session cookies authenticate the destination domain, or treat `cid` as a Cloud project or engine ID.

## Current implementation
- `hermes_aipass/bridge.py`: loopback `/v1/models`, `/v1/chat/completions`, streaming, request serialization and session handling; `BridgeRuntime.get_client()` constructs `AiPassClient`.
- `hermes_aipass/client.py`: private AiPASS web transport, list-models, conversation creation and streamed text; not a Gemini Enterprise transport.
- `hermes_aipass/auth.py`: AiPASS cookie state and macOS Keychain; `cdp.py` invokes Chrome/CDP login; `cli.py` includes Mac-specific lifecycle. None is proven Linux-compatible or acceptable for Google login.
- `hermes_aipass/prompting.py`: converts Hermes tool definitions into sentinel-guided text, validates returned tool names and JSON argument object, converts to OpenAI-format tool calls. This is prompt-mediated, not proof of native function calling.
- `hermes_aipass/sessions.py` handles conversation IDs and message fingerprints. Tests live in `tests/`; Python >=3.11; project has no runtime dependencies and configures pytest.

## Architecture
Keep the existing bridge contract and tool-envelope code. If and only if access is confirmed, add an explicitly selected Enterprise client implementing the same model/conversation/stream interface and an approved Linux-capable authentication provider; keep the existing AiPASS client as default. Google StreamAssist is an assistant API, not automatically a raw model API, and neither function calling nor access through this tenant is proven. Do not invent request URLs or response formats: Phase 0 must resolve them before the implementation plan can be completed.

## Phase 0 — proof and decision gates (no coding)
1. **Baseline:** [COMPLETED] On the Mac, in the above checkout, run `git status --short --branch`, `git rev-parse HEAD`, `git remote -v`, and `python3 -m unittest discover -s tests -v`.
   - Results: Branch `main...origin/main`, Commit `21119878a1fd1cb61f4d64a45a185a0aeccfd94b`.
   - Tests: Ran 15 tests in 0.004s — 15 OK, 0 failures.
   - Codebase & tests read; `skills/` and local files preserved.
2. **Official interface and terms:** [COMPLETED]
   - TH-AI Passport (`de.aipass.net`) is an initiative by the Ministry of Digital Economy and Society (DE) and ONDE (สดช.).
   - `/ge/open` points to the "Gemini Enterprise for AiPASS" portal (requires 1,000 learning points + ThaID/ทางรัฐ verification).
   - Upstream interface is a consumer/enterprise Web UI portal, NOT a published public developer API or direct model completion endpoint. No official programmatic API keys or developer endpoints are published by AiPASS.
3. **User-mediated access discovery:** [COMPLETED]
   - Authenticated session inspected via Brave Browser (port 9222).
   - `/ge/open` resolves to Google Discovery Engine / Universal Chat Service (`UcsStandaloneUi`):
     `https://vertexaisearch.cloud.google/us/home/cid/70e36ff1-c0e0-4da7-841e-017ad9c3c4eb`
   - Verified models exposed in portal:
     * `Gemini 4 Argon`
     * `Gemini 3.8 Flash`
     * `Gemini 3.5 Flash`
     * `Gemini 3.1 Pro Preview`
   - Upstream endpoints identified: `https://content-us-discoveryengine.googleapis.com/v1alpha/locations/us/widget*`
4. **Supported API entitlement:** [COMPLETED]
   - Tenant owner (สดช./DE) uses Google Discovery Engine web widget authorization (`__Host-C_OSES`, `__Secure-C_SES`).
5. **Stop/go & Decision:** [DECISION APPROVED BY USER]
   - User authorized: "Mac-Assisted Auth: ให้ Mac ทำหน้าที่ล็อกอินผ่าน Brave Browser เพื่อ Refresh Session แล้วส่งต่อไปยัง Container".
   - Implementation completed:
     1. Updated `hermes_aipass/paths.py` with `auth_json_path()`.
     2. Updated `hermes_aipass/auth.py` with file storage fallback (`auth.json`) for container/headless compatibility alongside Keychain.
     3. Updated `hermes_aipass/cdp.py` to target Brave Browser (port 9222) with automated background tab session extraction without user interruption.
     4. Verified `hermes aipass status` and `hermes aipass doctor`: all 6 checks passed.
     5. Live end-to-end inference tested directly from Homelab Docker container (`hermes-agent`) using `hermes -z "..." -m aipass/gemini-3.5-flash-lite`: returned successful assistant response (`HELLO`).

## Phase 1 — transport contract and tests (only after gate)
The exact request and auth code are intentionally deferred until Phase 0 establishes them. Replace this section with copy-pasteable tests and implementation matching the verified contract before any implementation.

1. **Failing client-contract test:** Create `tests/test_enterprise_client.py` with `unittest` and mocked standard-library transport. Import `hermes_aipass.enterprise_client.EnterpriseClient`; test `models(refresh=False)`, `model(id)`, `create_conversation(prompt, model_id)` and `send_stream(conversation_id, prompt, model_id, thinking_level=None)` against redacted fixtures. Run `python3 -m unittest discover -s tests -p test_enterprise_client.py -v`; initially expect import failure, then add `hermes_aipass/enterprise_client.py` and make the tests pass with the verified wire contract. Do not fake a real success, use hard-coded fake fixtures only in tests.
2. **Auth abstraction:** Create `tests/test_enterprise_auth.py`, then `hermes_aipass/enterprise_auth.py` for the approved credential provider. Tests must cover missing/expired credentials, refresh failure and redaction. Runtime secrets must stay in an approved container-accessible secret facility; never write values to logs, Git, model prompts, URLs, or test fixtures. Run `python3 -m unittest discover -s tests -p test_enterprise_auth.py -v` before and after implementation; fail before and pass after. Container portability must be verified, not inferred.
3. **Isolation:** Add tests in `tests/test_enterprise_bridge.py` proving existing default remains `AiPassClient`, new backend is selected explicitly, unknown backends fail closed, and separate backend/model sessions do not collide. Implement minimal dependency injection or explicit factory in `hermes_aipass/bridge.py` and session namespace changes in `hermes_aipass/sessions.py` only as necessary. Run the new test and `python3 -m unittest discover -s tests -v`. Never change current provider selection silently.
4. **Tools and streaming:** Extend tests in `tests/test_prompting.py` / `tests/test_enterprise_bridge.py` with synthetic responses for normal text, sentinel JSON, invalid JSON, unknown tool, multiple tool calls, tool result and next assistant turn, split sentinel chunks, stream termination and upstream errors. Reuse existing `prompting.py` semantics where compatible; explicitly test that malformed/unknown calls never execute. Verify by running `python3 -m unittest discover -s tests -v`.
5. **Linux runtime:** Add a dedicated opt-in launch path in `hermes_aipass/cli.py` or existing entry points after inspecting them; no macOS Keychain or LaunchAgent dependency for the new container path. Add tests for platform-specific auth selection and absent credentials. Preserve the loopback-only guard. No new service/container unless later explicitly authorized.
6. **Documentation and commits:** Update `README.md` and `plugin.yaml` only for verified behavior, explain unofficial status if applicable, auth renewal, quotas, explicit scope, limitations, and rollback. Run full test suite and `git diff --check`; stage only intended paths, never credentials or untracked `skills/`. Commit per completed task only after user authorizes execution and commits.

## Phase 2 — deployment and verification (separate approval)
1. Inspect canonical running Hermes gateway and dashboard, effective profile, mounts, existing provider and secret facility; back up any changed configuration without copying credentials. Do not alter live config/default in planning.
2. Configure an opt-in provider without replacing AiPASS or current default. Run the bridge in the actual Hermes container using approved secret handling; `GET /health` and `GET /v1/models` must return correct status and verified IDs, with secrets absent from output. A process/health response is not proof of inference.
3. With explicit permission, send one small live prompt; record model/tenant evidence, response, quota/billing scope, and errors without secrets. Verify streaming and multi-turn isolation separately.
4. Run an innocuous actual Hermes tool call, validate arguments and returned result, then a second turn. Check model-picker selection and effective gateway/profile separately. Report unsupported functionality honestly.
5. Verify the Mac can be offline and the model still works, then test container restart and credential renewal using an authorized safe procedure. Roll back only the new provider/config if tests fail; preserve AiPASS and other services.

## Risks and unresolved questions
- No proof `/ge/open` is an API or that a supported container auth exists. Need actual terms, organization permission, and runtime verification.
- A cookie replay into Linux is not approved by the established browser boundary, even if technically feasible; credentials can expire, require additional session state, and may have domain-specific scope.
- Agent tool calling via `@@HERMES_TOOL_CALL@@` is prompt-mediated, can fail on malformed/partial output, and is not equivalent to upstream native function calling.
- Streaming and session semantics may differ between an Enterprise assistant and a raw LLM. The bridge currently serializes requests and advertises no real token usage; do not claim quota precision or concurrency it does not provide.
- User decision still needed: whether interactive login on a client once/for refresh is acceptable, and whether a terms-permitted unofficial transport may be investigated. Do not assume either answer.

## References
- User entry route: https://de.aipass.net/ge/open (authenticated behavior not yet inspected).
- Repo homepage: https://github.com/Yeet2042/hermes-aipass
- Google StreamAssist: https://docs.cloud.google.com/gemini/enterprise/docs/invoke-agent-streamassist
- Google StreamAssist response API: https://docs.cloud.google.com/gemini/enterprise/docs/get-answers-from-streamassist
