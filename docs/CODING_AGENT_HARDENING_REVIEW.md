# Coding Agent Hardening Review

This document provides a read-only architecture study of mature open-source coding-agent repositories (`claude-code`, `SWE-agent`, `aider`, and `OpenHands`), comparing their approaches against our current local coding-agent harness. The goal is to identify concrete engineering patterns that can make our harness more reliable and capable.

---

## A. Repository-by-repository findings

### 1. anthropics/claude-code
* **Architecture:** Agentic CLI operating as a multi-layered harness (Input, Knowledge, Execution). 
* **Agent loop:** Asynchronous multi-turn loop. Evaluates prompts, invokes tools sequentially (or in parallel for safe tasks), and updates a context compressor.
* **Tool system:** Bash, str_replace, glob, grep, read, write. Strongly typed registry.
* **Editing:** Typically uses a search-and-replace file editing tool.
* **Verification:** Relies on bash tool to run tests and captures outputs.
* **Repair:** Uses a context compressor to manage long trajectories when errors compound.
* **Execution:** Local host execution but uses strict safety prompts and permission gating.
* **Observability:** Rich TUI/terminal output showing tools invoked and their status.
* **Relevant lessons:** The "Knowledge Layer" approach (e.g. `CLAUDE.md`) is excellent for enforcing project-specific rules.

### 2. SWE-agent/SWE-agent
* **Architecture:** Task-oriented agent designed for SWE-bench issues. Highly isolated runtime.
* **Agent loop:** Standard reasoning loop (Thought -> Action -> Observation).
* **Tool system:** Interacts with a pseudo-terminal. Uses bash commands wrapped as custom tools.
* **Editing:** Employs an interactive "editor" interface. The model issues commands like `open`, `search`, `scroll`, `edit`, and `submit`.
* **Verification:** Executes test scripts directly in the environment and reads the terminal output.
* **Repair:** Terminal feedback is appended directly to the context. 
* **Execution:** Strict separation between agent logic (host) and execution environment (Docker Sandbox).
* **Observability:** Saves full trajectories as `.jsonl` files for replay and debugging.
* **Relevant lessons:** An interactive editor interface reduces the need for the model to perfectly reproduce large chunks of code, bypassing whitespace/indentation diffing issues.

### 3. Aider-AI/aider
* **Architecture:** Interactive conversational REPL for pair programming.
* **Agent loop:** Interactive dialogue. Model can ask for clarity, and user can steer.
* **Tool system:** Uses edit formats and auto-commits rather than abstract JSON tools.
* **Editing:** Famous for adaptable edit formats (`udiff`, `whole`, `architect`/search-replace). The model generates `<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE` blocks.
* **Verification:** Runs linters and tests (if configured). Linter errors are automatically fed back to the model as follow-up prompts.
* **Repair:** Feeds lint and test errors back with a prompt asking the model to fix them.
* **Execution:** Operates directly on the user's local filesystem.
* **Observability:** Prints beautiful, colorized diffs to the terminal.
* **Relevant lessons:** AST-based "Repo Map" (via tree-sitter) provides context without reading all files. The SEARCH/REPLACE block strategy is highly effective for LLMs.

### 4. OpenHands/OpenHands
* **Architecture:** Event-driven state machine architecture separating agent frontend/backend from the runtime.
* **Agent loop:** EventStream processes `Observation` and `Action` objects.
* **Tool system:** Provides specific plugins (e.g., Jupyter, bash, file editor).
* **Editing:** Uses specialized file replacement and diff tools.
* **Verification:** Depends on bash actions for execution and verification.
* **Repair:** Feeds terminal output back into the event stream.
* **Execution:** Strictly sandboxed inside Docker via a Runtime interface.
* **Observability:** Robust React UI and trajectory logging.
* **Relevant lessons:** Strict `Action` and `Observation` schemas improve reliability and make the system easier to test.

---

## B. Cross-project patterns

1. **Repo Mapping / Context Curation:** Mature agents don't rely entirely on blind `read_file` or `grep`. They use AST-based repository maps (like Aider) or context compressors (like Claude Code) to give the model a high-level view of the codebase.
2. **Robust Edit Formats:** Simple string matching (`old_str`/`new_str`) is brittle. Mature agents use block-based Search/Replace (Aider), line-number based patching, or an interactive file-scrolling API (SWE-agent).
3. **Execution Separation:** SWE-agent and OpenHands explicitly separate the agent loop from the environment (e.g., Docker). Our Colab architecture mimics this conceptually, but lacks the robust I/O streaming of a true sandbox.
4. **Automated Linter Feedback:** Aider and others auto-run linters (like `flake8` or `ruff`) on modified files and automatically feed errors back before declaring success.
5. **Safety Guardrails:** All mature agents (even CLI ones like Aider/Claude Code) have safeguards for dangerous shell commands (e.g. `rm`, `git push`), requiring user confirmation.

---

## C. Our current gaps

| Area | Current implementation | Observed mature pattern | Gap | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Editing** | Strict string matching (`old_str`, `new_str`) | SEARCH/REPLACE blocks, `udiff`, or interactive editor | Our patch tool is brittle to whitespace and indentation changes | critical |
| **Repo Context** | Blind `read_file` or `run_shell` (`ls`, `grep`) | AST-based Repo Map (Tree-sitter) | Model wastes steps exploring or hallucinates structure | high |
| **Verification** | `run_tests` tool | Auto-linting and AST validation before test execution | Syntax errors crash the loop late rather than early | medium |
| **Safety** | Unrestricted `run_shell` | Permission gating / command blocking | Model could run destructive commands without confirmation | high |
| **Observability** | Simple print statements | Trajectory `.jsonl` logging and colored terminal diffs | Hard to debug complex failures; poor UX | medium |

---

## D. Recommended hardening

### Implement now

**1. Aider-style SEARCH/REPLACE Edit Tool**
* **Problem:** Our `write_patch` tool frequently fails due to exact string matching (especially whitespace/indentation).
* **Evidence:** Aider's `architect` format uses block-based search/replace, which is the industry standard for LLM code editing.
* **Benefit:** Dramatically higher edit success rate.
* **Complexity:** Medium (requires updating the tool parser and patching logic).
* **Risk:** Low.

**2. Trajectory Logging**
* **Problem:** Debugging agent failures requires reading standard output.
* **Evidence:** SWE-agent and OpenHands save strict `.jsonl` event trajectories.
* **Benefit:** Easier debugging and ability to build replay tools later.
* **Complexity:** Low (just append `AgentState` messages to a file).
* **Risk:** Low.

### Implement later

**1. AST-based Repo Map Context**
* **Problem:** The model wastes tokens and steps calling `read_file` just to understand project structure.
* **Evidence:** Aider's tree-sitter repo map is cited as a primary reason for its high SWE-bench score.
* **Benefit:** Gives the model instantaneous knowledge of classes and functions across the repo.
* **Complexity:** High (requires integrating `tree-sitter` or similar parsing libraries).
* **Risk:** Medium (could bloat the system prompt if not compressed properly).

**2. Automated Syntax/Linter Feedback**
* **Problem:** The model writes invalid Python, and we wait for `run_tests` to fail.
* **Evidence:** Aider automatically runs `ast.parse` or `flake8` on edits.
* **Benefit:** Fast, cheap feedback loop for syntax errors.
* **Complexity:** Medium.
* **Risk:** Low.

### Do not implement

**1. Docker Sandboxing**
* **Problem it solves:** Process isolation.
* **Evidence:** OpenHands/SWE-agent use Docker.
* **Why not to implement:** We already have the Colab GPU environment acting as our remote runtime. Adding Docker would violate our minimal dependency principles and add unnecessary complexity for Phase 1.

**2. Multi-Agent Orchestration / LangChain**
* **Problem it solves:** Task delegation.
* **Evidence:** Claude Code supports subagents.
* **Why not to implement:** Violates project principles. A single robust loop with good tools is sufficient for local coding tasks. Frameworks like LangChain add abstraction bloat.
