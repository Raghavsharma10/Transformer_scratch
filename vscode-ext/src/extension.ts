/**
 * vscode-ext/src/extension.ts
 *
 * Python Code Suggester VS Code Extension
 *
 * Features:
 *   - Debounced auto-trigger on every Python file change
 *   - Top-3 suggestions shown as inline ghost text decorations
 *   - Tab accepts suggestion 1, Alt+2 / Alt+3 for others
 *   - Status bar item shows current state (loading / ready / error)
 *   - Configurable via VS Code settings
 */

import * as vscode from "vscode";
import axios, { AxiosError } from "axios";

// ──────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────

interface SuggestResponse {
  suggestions: string[];
  scores:      number[];
  valid:       boolean[];
  latency_ms:  number;
}

interface SuggestionState {
  suggestions: string[];
  prefix:      string;
  position:    vscode.Position;
}

// ──────────────────────────────────────────────
// Decoration types (ghost text per suggestion)
// ──────────────────────────────────────────────

const GHOST_DECORATION_TYPES = [1, 2, 3].map((n) =>
  vscode.window.createTextEditorDecorationType({
    after: {
      fontStyle: "italic",
      color: new vscode.ThemeColor("editorGhostText.foreground"),
    },
  })
);

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────

let debounceTimer: ReturnType<typeof setTimeout> | undefined;
let currentState: SuggestionState | null = null;
let statusBar: vscode.StatusBarItem;
let enabled = true;

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

function getConfig() {
  const cfg = vscode.workspace.getConfiguration("codeSuggest");
  return {
    serverUrl:        cfg.get<string>("serverUrl",  "http://127.0.0.1:8000"),
    debounceMs:       cfg.get<number>("debounceMs", 400),
    maxTokens:        cfg.get<number>("maxTokens",  64),
    numSuggestions:   cfg.get<number>("numSuggestions", 3),
    triggerOnNewLine: cfg.get<boolean>("triggerOnNewLine", true),
    enabled:          cfg.get<boolean>("enabled", true),
  };
}

function clearDecorations(editor: vscode.TextEditor) {
  GHOST_DECORATION_TYPES.forEach((dt) => editor.setDecorations(dt, []));
  currentState = null;
  vscode.commands.executeCommand("setContext", "codeSuggest.hasSuggestions", false);
}

function showSuggestions(
  editor:      vscode.TextEditor,
  suggestions: string[],
  position:    vscode.Position
) {
  clearDecorations(editor);
  if (!suggestions.length) return;

  // Show first suggestion as primary ghost text after the cursor
  const lines = suggestions[0].split("\n");
  const primaryLine = lines[0];

  const decoration0: vscode.DecorationOptions = {
    range: new vscode.Range(position, position),
    renderOptions: {
      after: {
        contentText: primaryLine + (lines.length > 1 ? " ↵ …" : "") +
                     (suggestions.length > 1
                       ? `   [Alt+2: alt · Alt+3: alt]`
                       : ""),
        color: new vscode.ThemeColor("editorGhostText.foreground"),
      },
    },
  };
  editor.setDecorations(GHOST_DECORATION_TYPES[0], [decoration0]);

  // Show hint lines for suggestions 2 & 3 as subsequent ghost lines
  suggestions.slice(1).forEach((sug, i) => {
    const hint = sug.split("\n")[0].slice(0, 60);
    const pos  = new vscode.Position(position.line + 1 + i, 0);
    const dec: vscode.DecorationOptions = {
      range: new vscode.Range(pos, pos),
      renderOptions: {
        before: {
          contentText: `${i + 2}. ${hint}`,
          color: new vscode.ThemeColor("editorGhostText.foreground"),
          fontStyle: "italic",
        },
      },
    };
    editor.setDecorations(GHOST_DECORATION_TYPES[i + 1], [dec]);
  });

  vscode.commands.executeCommand("setContext", "codeSuggest.hasSuggestions", true);
}

function setStatus(text: string, tooltip?: string) {
  statusBar.text     = `$(sparkle) ${text}`;
  statusBar.tooltip  = tooltip ?? text;
  statusBar.show();
}

// ──────────────────────────────────────────────
// Core: fetch suggestions from the local server
// ──────────────────────────────────────────────

async function fetchSuggestions(document: vscode.TextDocument, position: vscode.Position) {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document !== document) return;

  const cfg    = getConfig();
  const prefix = document.getText(new vscode.Range(new vscode.Position(0, 0), position));

  // Don't trigger on very short or whitespace-only prefixes
  if (prefix.trim().length < 10) return;

  setStatus("thinking…");

  try {
    const resp = await axios.post<SuggestResponse>(
      `${cfg.serverUrl}/suggest`,
      {
        prefix:     prefix,
        k:          cfg.numSuggestions,
        max_tokens: cfg.maxTokens,
      },
      { timeout: 5000 }
    );

    const { suggestions } = resp.data;
    if (!suggestions?.length) {
      setStatus("Python Suggester");
      return;
    }

    // Verify cursor hasn't moved
    const cur = editor.selection.active;
    if (cur.line !== position.line || cur.character !== position.character) return;

    currentState = { suggestions, prefix, position };
    showSuggestions(editor, suggestions, position);
    setStatus(`${suggestions.length} suggestion${suggestions.length > 1 ? "s" : ""} · Tab to accept`,
              `Latency: ${resp.data.latency_ms}ms`);
  } catch (err) {
    const ae = err as AxiosError;
    const msg = ae.code === "ECONNREFUSED"
      ? "Server offline"
      : `Error: ${ae.message}`;
    setStatus(msg);
    // Quietly swallow — don't interrupt the user
  }
}

// ──────────────────────────────────────────────
// Accept a specific suggestion by index
// ──────────────────────────────────────────────

async function acceptSuggestion(index: number) {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !currentState) return;

  const suggestion = currentState.suggestions[index];
  if (!suggestion) return;

  clearDecorations(editor);
  setStatus("Python Suggester");

  // Insert the completion at the current cursor position
  const cursorPos = editor.selection.active;
  await editor.edit((editBuilder) => {
    editBuilder.insert(cursorPos, suggestion);
  });

  // Move cursor to end of inserted text
  const newText   = editor.document.getText();
  const insertEnd = editor.document.positionAt(
    editor.document.offsetAt(cursorPos) + suggestion.length
  );
  editor.selection = new vscode.Selection(insertEnd, insertEnd);
}

// ──────────────────────────────────────────────
// Extension activate / deactivate
// ──────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
  // Status bar
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.command = "codeSuggest.toggle";
  setStatus("Python Suggester");
  context.subscriptions.push(statusBar);

  // ── Commands ──

  context.subscriptions.push(
    vscode.commands.registerCommand("codeSuggest.toggle", () => {
      enabled = !enabled;
      const editor = vscode.window.activeTextEditor;
      if (editor) clearDecorations(editor);
      setStatus(enabled ? "Python Suggester" : "Python Suggester (off)");
      vscode.window.showInformationMessage(
        `Python Code Suggester ${enabled ? "enabled" : "disabled"}.`
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codeSuggest.suggest", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.document.languageId !== "python") return;
      fetchSuggestions(editor.document, editor.selection.active);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codeSuggest.acceptSuggestion1", () =>
      acceptSuggestion(0)
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("codeSuggest.acceptSuggestion2", () =>
      acceptSuggestion(1)
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("codeSuggest.acceptSuggestion3", () =>
      acceptSuggestion(2)
    )
  );

  // ── Auto-trigger on document change ──

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      if (event.document !== editor.document) return;
      if (editor.document.languageId !== "python") return;
      if (!enabled || !getConfig().enabled) return;

      // Clear stale decorations immediately on any change
      clearDecorations(editor);

      // Debounce the API call
      if (debounceTimer) clearTimeout(debounceTimer);

      const cfg = getConfig();

      // Immediately trigger on newline if configured
      const lastChange = event.contentChanges[event.contentChanges.length - 1];
      const isNewLine  = lastChange?.text.includes("\n");
      const delay      = isNewLine && cfg.triggerOnNewLine ? 100 : cfg.debounceMs;

      debounceTimer = setTimeout(() => {
        const pos = editor.selection.active;
        fetchSuggestions(event.document, pos);
      }, delay);
    })
  );

  // ── Clear suggestions when cursor moves away ──

  context.subscriptions.push(
    vscode.window.onDidChangeTextEditorSelection((event) => {
      if (currentState && event.textEditor.document.languageId === "python") {
        const cur = event.selections[0].active;
        if (!cur.isEqual(currentState.position)) {
          clearDecorations(event.textEditor);
        }
      }
    })
  );

  // ── Clear when switching documents ──

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor) clearDecorations(editor);
      currentState = null;
    })
  );

  console.log("Python Code Suggester extension activated.");
}

export function deactivate() {
  if (debounceTimer) clearTimeout(debounceTimer);
  GHOST_DECORATION_TYPES.forEach((dt) => dt.dispose());
  statusBar?.dispose();
}
