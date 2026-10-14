# FHGR HackApertus Template 2026

This repository contains the basic setup for the FHGR Hackapertus Challenge 2026 with Aegra and the Agent Chat UI.

## Getting Started

This template uses the following dependencies:

- [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui)
- [Aegra CLI](https://github.com/aegra/aegra)

### 1. Install Agent Chat UI

Clone the Agent Chat UI repository into a **separate directory** from this project:

```bash
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
```

> **Note:** Do not use `npx` to install Agent Chat UI, as it currently installs an older version.

Install the dependencies:

```bash
pnpm install
```

Start the development server:

```bash
pnpm dev
```

### 2. Start Aegra

In a separate terminal, navigate to this project's directory and start the Aegra development server:

```bash
uv run aegra dev
```

Open the Agent Chat UI in your browser, change the port to `2026`, and click **Continue** to connect to your agent.

## Production

When you're ready to run Aegra in production mode, use:

```bash
uv run aegra up
```
