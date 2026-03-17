#!/usr/bin/env python3
"""Generate an HTML report from Claude Code benchmark results."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime


# Keep in sync with run_claude_code.py
DISGUISED_TASKS = {"Apple--17"}
DISGUISED_GROUPS = {"Google Search"}
DISGUISE_TOOLTIP = "This test used undetected-chromedriver to avoid bot detection."


def extract_tool_responses_and_thinking(
    transcript_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Extract tool_use_id -> response text and tool key -> thinking text from a transcript."""
    responses = {}
    thinking_by_tool_key = {}  # tool key -> thinking text
    pending_tool_ids = {}  # tool_use_id -> tool_name(input) string
    pending_thinking = None  # thinking text waiting to be assigned to a tool call

    with open(transcript_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") == "assistant":
                content = entry.get("message", {}).get("content", [])
                block_types = {b.get("type") for b in content if isinstance(b, dict)}

                if block_types == {"thinking"}:
                    # Standalone thinking turn — save for the next tool call
                    for block in content:
                        if block.get("type") == "thinking" and block.get("thinking"):
                            pending_thinking = block["thinking"]
                            break
                else:
                    for block in content:
                        if block.get("type") == "tool_use" and block.get("id"):
                            tool_name = block.get("name", "unknown")
                            tool_input = json.dumps(block.get("input", {}))
                            key = f"{tool_name}({tool_input})"
                            pending_tool_ids[block["id"]] = key
                            if pending_thinking:
                                thinking_by_tool_key[key] = pending_thinking
                                pending_thinking = None

            elif entry.get("type") == "user":
                msg_content = entry.get("message", {}).get("content", [])
                if isinstance(msg_content, list):
                    for block in msg_content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                        ):
                            tid = block.get("tool_use_id")
                            if tid and tid in pending_tool_ids:
                                result_text = ""
                                for rc in block.get("content", []):
                                    if (
                                        isinstance(rc, dict)
                                        and rc.get("type") == "text"
                                    ):
                                        result_text += rc.get("text", "")
                                if result_text:
                                    responses[pending_tool_ids[tid]] = result_text
                                del pending_tool_ids[tid]

    return responses, thinking_by_tool_key


def enrich_messages_with_tool_responses(
    messages: list, tool_responses: dict[str, str], thinking_by_tool_key: dict[str, str]
) -> None:
    """Attach tool_response and thinking to messages that match tool call patterns."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content", "").startswith(
            "Tool:"
        ):
            content = msg["content"]
            if content in tool_responses and "tool_response" not in msg:
                msg["tool_response"] = tool_responses[content]
            if content in thinking_by_tool_key and "thinking" not in msg:
                msg["thinking"] = thinking_by_tool_key[content]


def load_task_data(task_dir: Path) -> dict | None:
    """Load all data for a single task."""
    metadata_path = task_dir / "metadata.json"
    messages_path = task_dir / "interact_messages.json"
    transcript_path = task_dir / "transcript.jsonl"
    eval_path = task_dir / "eval.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    messages = []
    if messages_path.exists():
        with open(messages_path) as f:
            messages = json.load(f)

    # Enrich messages with tool responses and thinking from transcript if missing
    has_tool_responses = any(m.get("tool_response") for m in messages)
    if transcript_path.exists():
        try:
            tool_responses, thinking_by_tool_key = extract_tool_responses_and_thinking(
                transcript_path
            )
            if not has_tool_responses or thinking_by_tool_key:
                enrich_messages_with_tool_responses(
                    messages, tool_responses, thinking_by_tool_key
                )
        except Exception:
            pass

    # Find all screenshots
    screenshots = sorted(task_dir.glob("screenshot*.jpg")) + sorted(
        task_dir.glob("screenshot*.png")
    )
    # Sort by number
    screenshots = sorted(
        screenshots, key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)
    )

    eval_data = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)

    return {
        "name": task_dir.name,
        "metadata": metadata,
        "messages": messages,
        "screenshots": screenshots,
        "eval": eval_data,
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def format_tokens(tokens: int) -> str:
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on token usage.

    Pricing: $0.05 per 1M input tokens, $0.40 per 1M output tokens.
    """
    input_cost = (input_tokens / 1_000_000) * 0.05
    output_cost = (output_tokens / 1_000_000) * 0.40
    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format cost in dollars with 2 decimal places."""
    return f"${cost:.2f}"


def format_cost_cents(cost: float) -> str:
    """Format cost in cents."""
    return f"\u00a2{cost * 100:.2f}"


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("✅", "✓")
        .replace("❌", "✗")
    )


def thinking_to_html(thinking: str) -> str:
    """Convert a thinking block to a standalone message HTML."""
    escaped_thinking = escape_html(thinking).replace("\n", "<br>")
    return f"""
    <details class="message thinking-block" open>
        <summary class="message-role thinking-label">thinking</summary>
        <div class="thinking-content">{escaped_thinking}</div>
    </details>
    """


def message_to_html(message: dict) -> str:
    """Convert a message to HTML."""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    tool_response = message.get("tool_response", "")

    role_class = {"system": "system", "user": "user", "assistant": "assistant"}.get(
        role, "unknown"
    )

    role_label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
        role, role.title()
    )

    # Check if it's a tool call
    is_tool = content.startswith("Tool:")
    if is_tool:
        role_class = "tool"
        role_label = "Tool Call"
        content = content.removeprefix("Tool:").strip()

    escaped_content = escape_html(content)
    # Convert newlines to <br> for display
    escaped_content = escaped_content.replace("\n", "<br>")

    # Add inline tool response if present
    tool_response_html = ""
    if tool_response:
        escaped_response = escape_html(tool_response)
        tool_response_html = f"""
        <div class="tool-response">
            <div class="tool-response-label">Response</div>
            <pre class="tool-response-content">{escaped_response}</pre>
        </div>
        """

    return f"""
    <div class="message {role_class}">
        <div class="message-role">{role_label}</div>
        <div class="message-content">{escaped_content}</div>
        {tool_response_html}
    </div>
    """


def generate_task_html(task: dict, task_index: int, disguised: bool = False) -> str:
    """Generate HTML for a single task."""
    metadata = task["metadata"]
    messages = task["messages"]
    screenshots = task["screenshots"]
    eval_data = task.get("eval")

    task_id = metadata.get("task_id", task["name"])
    question = metadata.get("question", "N/A")
    start_url = metadata.get("start_url", "N/A")
    final_answer = metadata.get("final_answer", "N/A")
    duration = metadata.get("duration_seconds", 0)
    costs = metadata.get("costs", {})

    # Calculate total tokens
    alumnium_costs = costs.get("alumnium", {}).get("total", {})
    total_tokens = alumnium_costs.get("total_tokens", 0)
    input_tokens = alumnium_costs.get("input_tokens", 0)
    output_tokens = alumnium_costs.get("output_tokens", 0)

    # Screenshots gallery - use relative paths from report location
    screenshots_html = ""
    if screenshots:
        screenshots_html = '<div class="screenshots-gallery">'
        for i, screenshot in enumerate(screenshots):
            rel_path = f"{task['name']}/{screenshot.name}"
            screenshots_html += f"""
            <div class="screenshot-item">
                <img src="{rel_path}" alt="Screenshot {i + 1}" loading="lazy" onclick="window.open(this.src, '_blank')">
                <div class="screenshot-label">Step {i + 1}</div>
            </div>
            """
        screenshots_html += "</div>"
    else:
        screenshots_html = '<p class="no-screenshots">No screenshots available</p>'

    # Messages/conversation
    messages_html = ""
    if messages:
        for msg in messages:
            if msg.get("thinking"):
                messages_html += thinking_to_html(msg["thinking"])
            messages_html += message_to_html(msg)
    else:
        messages_html = '<p class="no-messages">No conversation history available</p>'

    # Final answer - convert markdown-ish to HTML
    final_answer_html = escape_html(final_answer).replace("\n", "<br>")

    # Eval badge
    if eval_data is None:
        eval_badge = ""
        task_class = "task"
    elif eval_data.get("result") == 1:
        eval_badge = '<span class="eval-badge success">SUCCESS</span>'
        task_class = "task eval-success"
    elif eval_data.get("result") == 0:
        eval_badge = '<span class="eval-badge failure">FAILURE</span>'
        task_class = "task eval-failure"
    else:
        eval_badge = '<span class="eval-badge unknown">[ UNKNOWN ]</span>'
        task_class = "task"

    # Eval section
    eval_section_html = ""
    if eval_data:
        escaped_response = escape_html(eval_data.get("response", "")).replace(
            "\n", "<br>"
        )
        eval_section_html = f"""
            <div class="section">
                <h3>auto eval</h3>
                <div class="eval-response">{escaped_response}</div>
            </div>
        """

    return f"""
    <div class="{task_class}" id="task-{task_index}">
        <div class="task-header" onclick="toggleTask({task_index})">
            <div class="task-title">
                <span class="task-toggle">▶</span>
                <h2>{escape_html(task_id)}</h2>
                {eval_badge}
                {'<span class="disguise-badge" title="' + DISGUISE_TOOLTIP + '">(-_-)</span>' if disguised else ""}
            </div>
            <div class="task-meta">
                <span class="meta-item duration">◷ {format_duration(duration)}</span>
                <span class="meta-item tokens">⊞ {format_tokens(total_tokens)} tokens</span>
            </div>
        </div>
        <div class="task-body" style="display: none;">
            <div class="task-info">
                <div class="info-row">
                    <strong>Question:</strong>
                    <span>{escape_html(question)}</span>
                </div>
                <div class="info-row">
                    <strong>Start URL:</strong>
                    <a href="{escape_html(start_url)}" target="_blank">{escape_html(start_url)}</a>
                </div>
                <div class="info-row">
                    <strong>Token Usage:</strong>
                    <span>Input: {format_tokens(input_tokens)} | Output: {format_tokens(output_tokens)} | Total: {format_tokens(total_tokens)}</span>
                </div>
            </div>

            <div class="section">
                <h3>screenshots</h3>
                {screenshots_html}
            </div>

            <div class="section">
                <h3>conversation</h3>
                <div class="conversation">
                    {messages_html}
                </div>
            </div>

            <div class="section">
                <h3>final answer</h3>
                <div class="final-answer">
                    {final_answer_html}
                </div>
            </div>
            {eval_section_html}
        </div>
    </div>
    """


def get_task_group(task_name: str) -> str:
    """Extract website group name from task directory name.

    e.g. "taskAllrecipes--10" -> "Allrecipes"
    """
    name = task_name.removeprefix("task")
    parts = name.rsplit("--", 1)
    return parts[0]


def generate_group_html(group_name: str, group_tasks: list, start_index: int) -> str:
    """Generate HTML for a collapsible group of tasks."""
    evaluated = [t for t in group_tasks if t.get("eval") is not None]
    successful = [t for t in evaluated if t["eval"].get("result") == 1]

    if evaluated:
        rate = len(successful) / len(evaluated)
        meta_text = f"{rate:.0%} ({len(successful)}/{len(evaluated)})"
    else:
        meta_text = "N/A"

    is_disguised_group = group_name in DISGUISED_GROUPS
    tasks_html = ""
    for i, task in enumerate(group_tasks):
        task_short = task["name"].removeprefix("task")
        disguised = is_disguised_group or task_short in DISGUISED_TASKS
        tasks_html += generate_task_html(task, start_index + i, disguised=disguised)

    return f"""
    <div class="task-group" id="group-{start_index}">
        <div class="group-header" onclick="toggleGroup({start_index})">
            <div class="group-title">
                <span class="group-toggle">▶</span>
                <h2>{escape_html(group_name)}</h2>
            </div>
            <div class="group-meta">
                <span class="meta-item">{meta_text}</span>
            </div>
        </div>
        <div class="group-body" style="display: none;">
            {tasks_html}
        </div>
    </div>
    """


def generate_report(results_dir: Path, output_path: Path) -> None:
    """Generate the complete HTML report."""
    # Load all tasks
    tasks = []

    def task_sort_key(p):
        name = p.name  # e.g. "taskAllrecipes--10"
        parts = name.rsplit("--", 1)
        prefix = parts[0]
        suffix = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        return (prefix, suffix)

    for task_dir in sorted(results_dir.iterdir(), key=task_sort_key):
        if task_dir.is_dir() and task_dir.name.startswith("task"):
            task_data = load_task_data(task_dir)
            if task_data:
                tasks.append(task_data)

    # Calculate summary stats
    total_tasks = len(tasks)
    total_duration = sum(t["metadata"].get("duration_seconds", 0) for t in tasks)
    total_tokens = sum(
        t["metadata"]
        .get("costs", {})
        .get("alumnium", {})
        .get("total", {})
        .get("total_tokens", 0)
        for t in tasks
    )
    total_input_tokens = sum(
        t["metadata"]
        .get("costs", {})
        .get("alumnium", {})
        .get("total", {})
        .get("input_tokens", 0)
        for t in tasks
    )
    total_output_tokens = sum(
        t["metadata"]
        .get("costs", {})
        .get("alumnium", {})
        .get("total", {})
        .get("output_tokens", 0)
        for t in tasks
    )

    # Steps stats (MCP tool calls)
    def count_steps(task):
        return sum(
            1
            for m in task.get("messages", [])
            if m.get("role") == "assistant"
            and m.get("content", "").startswith("Tool: mcp__alumnium__")
        )

    total_steps = sum(count_steps(t) for t in tasks)

    # Eval stats
    evaluated_tasks = [t for t in tasks if t.get("eval") is not None]
    successful_tasks = [t for t in evaluated_tasks if t["eval"].get("result") == 1]
    success_rate = (
        len(successful_tasks) / len(evaluated_tasks) if evaluated_tasks else None
    )

    # Group tasks by website
    groups: dict[str, list] = {}
    for task in tasks:
        group = get_task_group(task["name"])
        groups.setdefault(group, []).append(task)

    # Generate tasks HTML grouped
    tasks_html = ""
    task_index = 0
    for group_name, group_tasks in groups.items():
        tasks_html += generate_group_html(group_name, group_tasks, task_index)
        task_index += len(group_tasks)

    # Generate complete HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alumnium WebVoyager</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        :root {{
            --bg-primary: #000000;
            --bg-secondary: #080808;
            --bg-tertiary: #0d0d0d;
            --text-primary: #ffffff;
            --text-secondary: #aaaaaa;
            --accent: #00d4ff;
            --border: #00d4ff;
            --border-dim: #003344;
            --success: #00ff41;
            --failure: #ff0044;
            --info: #7eb8ff;
            --warning: #ffaa00;
            --purple: #bf00ff;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0; }}
        }}

        @keyframes scanmove {{
            0% {{ background-position: 0 0; }}
            100% {{ background-position: 0 4px; }}
        }}

        @keyframes bootline {{
            0% {{ width: 0; opacity: 1; }}
            100% {{ width: 100%; opacity: 1; }}
        }}

        @keyframes flicker {{
            0%, 97%, 100% {{ opacity: 1; }}
            98% {{ opacity: 0.96; }}
            99% {{ opacity: 0.98; }}
        }}

        body {{
            font-family: 'VT323', 'Courier New', Courier, monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 20px;
            font-size: 18px;
        }}

        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 0, 0, 0.13) 2px,
                rgba(0, 0, 0, 0.13) 4px
            );
            z-index: 9999;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* ── Header ── */
        header {{
            text-align: center;
            padding: 40px 20px 35px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.15), inset 0 0 60px rgba(0, 0, 0, 0.8);
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        }}

        header::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: var(--accent);
            box-shadow: 0 0 12px var(--accent);
        }}

        .terminal-prompt {{
            font-family: 'VT323', monospace;
            color: var(--text-secondary);
            font-size: 1.1em;
            margin-bottom: 18px;
            opacity: 0.75;
            letter-spacing: 1px;
        }}

        header h1 {{
            font-family: 'Press Start 2P', monospace;
            font-size: 1.6em;
            line-height: 1.5;
            margin-bottom: 16px;
        }}

        header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.25em;
            letter-spacing: 2px;
        }}

        .cursor {{
            animation: blink 1s step-end infinite;
            color: var(--accent);
        }}

        /* ── Summary ── */
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}

        .summary-card {{
            background: var(--bg-secondary);
            padding: 22px 20px 18px;
            text-align: center;
            border: 1px solid var(--border);
            box-shadow: 0 0 12px rgba(0, 212, 255, 0.1), inset 0 0 20px rgba(0, 0, 0, 0.5);
            position: relative;
        }}

        .summary-card::before {{
            content: 'SUCCESS RATE';
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-primary);
            padding: 0 8px;
            font-size: 0.8em;
            color: var(--text-secondary);
            letter-spacing: 2px;
        }}

        .summary-card .value {{
            font-family: 'Press Start 2P', monospace;
            font-size: 1.3em;
            color: var(--accent);
            margin-bottom: 10px;
            word-break: break-all;
            line-height: 1.6;
        }}

        .summary-card .label {{
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 1em;
        }}

        /* ── Stats table ── */
        .stats-table {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            margin-bottom: 30px;
            overflow: hidden;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.08);
        }}

        .stats-row {{
            display: grid;
            grid-template-columns: 140px repeat(4, 1fr);
            border-bottom: 1px solid var(--border-dim);
        }}

        .stats-row:last-child {{
            border-bottom: none;
        }}

        .stats-header {{
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border) !important;
        }}

        .stats-header .stats-cell {{
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 1em;
        }}

        .stats-cell {{
            padding: 12px 20px;
            text-align: center;
            font-size: 1.15em;
        }}

        .stats-row-label {{
            text-align: left !important;
            color: var(--accent) !important;
            background: rgba(0, 212, 255, 0.04);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* ── Task groups ── */
        .task-group {{
            margin-bottom: 10px;
            border: 1px solid var(--border-dim);
        }}

        .group-header {{
            padding: 13px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-secondary);
            transition: all 0.12s;
        }}

        .task-group.expanded .group-header {{
            border-bottom: 1px solid var(--border-dim);
            background: rgba(0, 212, 255, 0.04);
        }}

        .group-header:hover {{
            background: rgba(0, 212, 255, 0.07);
            box-shadow: inset 3px 0 0 var(--accent);
        }}

        .group-title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .group-title h2 {{
            font-family: 'Press Start 2P', monospace;
            font-size: 0.78em;
            color: var(--text-primary);
            text-transform: uppercase;
            line-height: 1.6;
        }}

        .group-toggle {{
            color: var(--accent);
            font-size: 1.1em;
            transition: transform 0.2s;
        }}

        .task-group.expanded .group-toggle {{
            transform: rotate(90deg);
        }}

        .group-meta {{
            color: var(--text-secondary);
            font-size: 1.1em;
        }}

        .group-body {{
            padding: 12px;
            background: var(--bg-primary);
        }}

        /* ── Tasks ── */
        .task {{
            background: var(--bg-secondary);
            margin-bottom: 8px;
            border: 1px solid var(--border-dim);
            overflow: hidden;
        }}

        .task-header {{
            padding: 11px 18px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-tertiary);
            transition: all 0.12s;
        }}

        .task-header:hover {{
            background: rgba(0, 212, 255, 0.05);
            box-shadow: inset 3px 0 0 var(--accent);
        }}

        .task-title {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .task-title h2 {{
            font-size: 1.15em;
            font-weight: normal;
        }}

        .task-toggle {{
            color: var(--accent);
            transition: transform 0.2s;
        }}

        .task.expanded .task-toggle {{
            transform: rotate(90deg);
        }}

        .task-meta {{
            display: flex;
            gap: 20px;
        }}

        .meta-item {{
            font-size: 1em;
            color: var(--text-secondary);
        }}

        .task-body {{
            padding: 18px;
            border-top: 1px solid var(--border-dim);
        }}

        .task-info {{
            background: var(--bg-primary);
            padding: 14px;
            margin-bottom: 18px;
            border: 1px solid var(--border-dim);
            border-left: 3px solid var(--accent);
        }}

        .info-row {{
            margin-bottom: 8px;
            display: flex;
            gap: 10px;
            align-items: baseline;
        }}

        .info-row:last-child {{ margin-bottom: 0; }}

        .info-row strong {{
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
            white-space: nowrap;
            min-width: 130px;
        }}

        .info-row a {{
            color: var(--info);
            text-decoration: none;
        }}

        .info-row a:hover {{
            text-decoration: underline;
        }}

        /* ── Sections ── */
        .section {{
            margin-top: 20px;
        }}

        .section h3 {{
            margin-bottom: 12px;
            padding: 7px 14px;
            border-left: 3px solid var(--accent);
            background: rgba(0, 212, 255, 0.04);
            color: var(--text-primary);
            font-size: 1.15em;
            text-transform: uppercase;
            letter-spacing: 3px;
        }}

        /* ── Screenshots ── */
        .screenshots-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}

        .screenshot-item {{
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-dim);
            transition: border-color 0.15s, box-shadow 0.15s;
        }}

        .screenshot-item:hover {{
            border-color: var(--accent);
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.35);
        }}

        .screenshot-item img {{
            width: 100%;
            height: 150px;
            object-fit: cover;
            object-position: top;
            cursor: pointer;
            transition: transform 0.2s, filter 0.2s;
            filter: sepia(0.15) hue-rotate(85deg) saturate(0.75) brightness(0.9);
        }}

        .screenshot-item:hover img {{
            filter: none;
            transform: scale(1.02);
        }}

        .screenshot-label {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.88);
            padding: 4px;
            text-align: center;
            font-size: 0.9em;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-dim);
            letter-spacing: 1px;
        }}

        /* ── Conversation ── */
        .conversation {{
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            background: var(--bg-primary);
            border: 1px solid var(--border-dim);
        }}

        .message {{
            margin-bottom: 10px;
            padding: 10px 14px;
            border-left: 3px solid;
        }}

        .message.system {{
            background: rgba(0, 212, 255, 0.04);
            border-left-color: var(--accent);
        }}

        .message.user {{
            background: rgba(0, 212, 255, 0.04);
            border-left-color: var(--accent);
        }}

        .message.assistant {{
            background: rgba(0, 255, 65, 0.03);
            border-left-color: var(--success);
        }}

        .message.tool {{
            background: rgba(255, 170, 0, 0.04);
            border-left-color: var(--warning);
            font-family: 'VT323', 'Courier New', monospace;
        }}

        .message-role {{
            font-weight: normal;
            margin-bottom: 5px;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}

        .message.system .message-role {{ color: var(--accent); }}
        .message.user .message-role {{ color: var(--accent); }}
        .message.assistant .message-role {{ color: var(--success); }}
        .message.tool .message-role {{ color: var(--warning); }}

        .message-content {{ word-wrap: break-word; }}

        .final-answer {{
            background: var(--bg-primary);
            padding: 16px 20px;
            border: 1px solid var(--border-dim);
            border-left: 3px solid var(--success);
        }}

        .section:has(.final-answer) h3 {{
            border-left-color: var(--success);
            background: rgba(0, 255, 65, 0.04);
            color: var(--success);
        }}

        .section:has(.eval-response) h3 {{
            color: var(--accent);
        }}

        .no-screenshots, .no-messages {{
            color: var(--text-secondary);
            opacity: 0.6;
        }}

        .disguise-badge {{ cursor: help; font-size: 1.1em; }}

        /* ── Eval badges ── */
        .eval-badge {{
            display: inline-block;
            padding: 1px 10px;
            font-family: 'VT323', monospace;
            font-size: 1em;
            letter-spacing: 1px;
            border: 1px solid;
        }}

        .eval-badge.success {{
            background: rgba(0, 212, 255, 0.1);
            color: var(--success);
            border-color: var(--success);
        }}

        .eval-badge.failure {{
            background: rgba(255, 0, 68, 0.1);
            color: var(--failure);
            border-color: var(--failure);
        }}

        .eval-badge.unknown {{
            background: rgba(255, 170, 0, 0.1);
            color: var(--warning);
            border-color: var(--warning);
        }}

        .task.eval-success .task-header {{ border-left: 3px solid var(--success); }}
        .task.eval-failure .task-header {{ border-left: 3px solid var(--failure); }}

        .eval-response {{
            background: var(--bg-primary);
            padding: 14px;
            border: 1px solid var(--border-dim);
            border-left: 3px solid var(--accent);
            line-height: 1.7;
        }}

        /* ── Tool response ── */
        .tool-response {{
            margin-top: 8px;
            border-top: 1px solid rgba(255, 170, 0, 0.2);
            padding-top: 6px;
        }}

        .tool-response-label {{
            color: var(--warning);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }}

        .tool-response-content {{
            padding: 8px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 170, 0, 0.15);
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1em;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 300px;
            overflow-y: auto;
        }}

        /* ── Thinking block ── */
        .message.thinking-block {{
            background: rgba(191, 0, 255, 0.04);
            border-left-color: var(--purple);
        }}

        .thinking-label {{
            color: var(--purple);
            cursor: pointer;
            user-select: none;
            list-style: none;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}

        .thinking-label::-webkit-details-marker {{ display: none; }}

        .thinking-content {{
            margin-top: 8px;
            font-size: 0.95em;
            line-height: 1.6;
            word-break: break-word;
        }}

        /* ── Modal ── */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.96);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            overflow: auto;
        }}

        .modal.active {{ display: flex; }}

        .modal-image-container {{
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100%;
            padding: 60px 80px;
        }}

        .modal img {{
            max-width: 100%;
            max-height: calc(100vh - 120px);
            object-fit: contain;
            border: 1px solid var(--border);
            box-shadow: 0 0 40px rgba(0, 212, 255, 0.25);
        }}

        .modal-close {{
            position: fixed;
            top: 20px;
            right: 30px;
            font-size: 40px;
            color: var(--accent);
            cursor: pointer;
            z-index: 1001;
            font-family: 'VT323', monospace;
        }}

        .modal-nav {{
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            font-size: 56px;
            color: var(--accent);
            cursor: pointer;
            z-index: 1001;
            padding: 20px;
            user-select: none;
            opacity: 0.55;
            transition: opacity 0.15s;
            font-family: 'VT323', monospace;
        }}

        .modal-nav:hover {{ opacity: 1; }}

        .modal-nav.disabled {{
            opacity: 0.12;
            cursor: default;
        }}

        .modal-nav.prev {{ left: 10px; }}
        .modal-nav.next {{ right: 10px; }}

        .modal-counter {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: var(--text-secondary);
            font-size: 1.2em;
            z-index: 1001;
            background: rgba(0, 0, 0, 0.85);
            padding: 5px 18px;
            border: 1px solid var(--border-dim);
            letter-spacing: 3px;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-dim); }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--accent);
            box-shadow: 0 0 6px var(--accent);
        }}

        @media (max-width: 768px) {{
            header h1 {{ font-size: 1.1em; }}
            .task-header {{ flex-direction: column; gap: 10px; align-items: flex-start; }}
            .task-meta {{ flex-wrap: wrap; }}
            .stats-row {{ grid-template-columns: auto repeat(4, 1fr); }}
            .stats-cell {{ padding: 10px 8px; font-size: 0.95em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ALUMNIUM WEBVOYAGER</h1>
            <p class="subtitle">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<span class="cursor">_</span></p>
        </header>

        <div class="summary">
            <div class="summary-card">
                <div class="value">{"N/A" if success_rate is None else f"{success_rate:.1%} "}</div>
                <div class="label">{len(successful_tasks)}/{len(evaluated_tasks)}</div>
            </div>
        </div>

        <div class="stats-table">
            <div class="stats-row stats-header">
                <div class="stats-cell stats-row-label"></div>
                <div class="stats-cell">Duration</div>
                <div class="stats-cell">Tokens</div>
                <div class="stats-cell">Steps</div>
                <div class="stats-cell">Cost</div>
            </div>
            <div class="stats-row">
                <div class="stats-cell stats-row-label">Total</div>
                <div class="stats-cell">{format_duration(total_duration)}</div>
                <div class="stats-cell">{format_tokens(total_tokens)}</div>
                <div class="stats-cell">{total_steps}</div>
                <div class="stats-cell">{format_cost(calculate_cost(total_input_tokens, total_output_tokens))}</div>
            </div>
            <div class="stats-row">
                <div class="stats-cell stats-row-label">Average</div>
                <div class="stats-cell">{format_duration(total_duration / total_tasks) if total_tasks > 0 else "N/A"}</div>
                <div class="stats-cell">{format_tokens(int(total_tokens / total_tasks)) if total_tasks > 0 else "N/A"}</div>
                <div class="stats-cell">{f"{total_steps / total_tasks:.1f}" if total_tasks > 0 else "N/A"}</div>
                <div class="stats-cell">{format_cost_cents(calculate_cost(total_input_tokens, total_output_tokens) / total_tasks) if total_tasks > 0 else "N/A"}</div>
            </div>
        </div>

        <div class="tasks">
            {tasks_html}
        </div>
    </div>

    <!-- Modal for full-size screenshots with carousel -->
    <div class="modal" id="imageModal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <span class="modal-nav prev" id="modalPrev" onclick="navigateImage(-1)">&#8249;</span>
        <span class="modal-nav next" id="modalNext" onclick="navigateImage(1)">&#8250;</span>
        <div class="modal-image-container" id="modalContainer">
            <img id="modalImage" src="" alt="Full size screenshot">
        </div>
        <div class="modal-counter" id="modalCounter">1 / 1</div>
    </div>

    <script>
        // Task screenshots data
        const taskScreenshots = {{}};

        function toggleGroup(index) {{
            const group = document.getElementById('group-' + index);
            const body = group.querySelector('.group-body');
            const isExpanded = group.classList.contains('expanded');

            if (isExpanded) {{
                body.style.display = 'none';
                group.classList.remove('expanded');
            }} else {{
                body.style.display = 'block';
                group.classList.add('expanded');
            }}
        }}

        function toggleTask(index) {{
            const task = document.getElementById('task-' + index);
            const body = task.querySelector('.task-body');
            const isExpanded = task.classList.contains('expanded');

            if (isExpanded) {{
                body.style.display = 'none';
                task.classList.remove('expanded');
            }} else {{
                body.style.display = 'block';
                task.classList.add('expanded');
            }}
        }}

        // Carousel state
        let currentTaskIndex = -1;
        let currentImageIndex = 0;

        function collectTaskScreenshots() {{
            // Collect all screenshot URLs for each task
            document.querySelectorAll('.task').forEach((task, taskIndex) => {{
                const screenshots = [];
                task.querySelectorAll('.screenshot-item img').forEach(img => {{
                    screenshots.push(img.src);
                }});
                taskScreenshots[taskIndex] = screenshots;
            }});
        }}

        function updateModalUI() {{
            const screenshots = taskScreenshots[currentTaskIndex] || [];
            const total = screenshots.length;
            const current = currentImageIndex + 1;

            document.getElementById('modalCounter').textContent = `${{current}} / ${{total}}`;

            const prevBtn = document.getElementById('modalPrev');
            const nextBtn = document.getElementById('modalNext');

            prevBtn.classList.toggle('disabled', currentImageIndex <= 0);
            nextBtn.classList.toggle('disabled', currentImageIndex >= total - 1);
        }}

        function openModal(taskIndex, imageIndex) {{
            collectTaskScreenshots();
            currentTaskIndex = taskIndex;
            currentImageIndex = imageIndex;

            const screenshots = taskScreenshots[taskIndex] || [];
            if (screenshots.length === 0) return;

            const modal = document.getElementById('imageModal');
            const img = document.getElementById('modalImage');
            img.src = screenshots[imageIndex];
            updateModalUI();
            modal.classList.add('active');
            event.stopPropagation();
        }}

        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
            currentTaskIndex = -1;
            currentImageIndex = 0;
        }}

        function navigateImage(direction) {{
            const screenshots = taskScreenshots[currentTaskIndex] || [];
            const newIndex = currentImageIndex + direction;

            if (newIndex < 0 || newIndex >= screenshots.length) return;

            currentImageIndex = newIndex;
            document.getElementById('modalImage').src = screenshots[currentImageIndex];
            updateModalUI();
        }}

        // Close on background click (not on image/controls)
        document.getElementById('imageModal').addEventListener('click', function(e) {{
            if (e.target === this || e.target.id === 'modalContainer') closeModal();
        }});

        document.addEventListener('keydown', function(e) {{
            const modal = document.getElementById('imageModal');
            if (!modal.classList.contains('active')) return;

            if (e.key === 'Escape') closeModal();
            if (e.key === 'ArrowLeft') navigateImage(-1);
            if (e.key === 'ArrowRight') navigateImage(1);
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report generated: {output_path}")
    print(f"  - {total_tasks} tasks")
    print(f"  - {format_duration(total_duration)} total duration")
    print(f"  - {format_tokens(total_tokens)} total tokens")
    print(f"  - {format_cost(calculate_cost(total_input_tokens, total_output_tokens))} estimated cost")


def generate_standalone_report(results_dir: Path, output_dir: Path) -> None:
    """Generate a self-contained report folder with screenshots copied alongside the HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy screenshots for each task, preserving task/screenshot.ext structure
    for task_dir in results_dir.iterdir():
        if not (task_dir.is_dir() and task_dir.name.startswith("task")):
            continue
        screenshots = list(task_dir.glob("screenshot*.jpg")) + list(
            task_dir.glob("screenshot*.png")
        )
        if not screenshots:
            continue
        dest_task_dir = output_dir / task_dir.name
        dest_task_dir.mkdir(exist_ok=True)
        for screenshot in screenshots:
            shutil.copy2(screenshot, dest_task_dir / screenshot.name)

    output_path = output_dir / "index.html"
    generate_report(results_dir, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from benchmark results."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/claude-code",
        help="Path to results directory (default: results/claude-code)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Create a self-contained report folder with screenshots copied alongside the HTML.",
    )
    parser.add_argument(
        "--output",
        help="Output path. For --standalone, this is the folder to create (default: <results_dir>-standalone). Otherwise, the HTML file path (default: <results_dir>/report.html).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if args.standalone:
        output_dir = (
            Path(args.output)
            if args.output
            else results_dir.parent / (results_dir.name + "-standalone")
        )
        generate_standalone_report(results_dir, output_dir)
    else:
        output_path = Path(args.output) if args.output else results_dir / "report.html"
        generate_report(results_dir, output_path)


if __name__ == "__main__":
    main()
