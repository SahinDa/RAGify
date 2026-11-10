const ICON_COPY = `<svg viewBox="0 0 24 24"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>`;
const ICON_CHECK = `<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

let conversationId = localStorage.getItem("ragify_conv_id");
if (!conversationId) {
  conversationId =
    "conv_" +
    Math.random().toString(36).substring(2, 9) +
    Date.now().toString(36);
  localStorage.setItem("ragify_conv_id", conversationId);
}

// --- Theme Management ---
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("ragify_theme", theme);

  const iconEl = document.getElementById("themeIcon");
  const labelEl = document.getElementById("themeLabel");
  if (theme === "light") {
    if (iconEl) iconEl.textContent = "🌙";
    if (labelEl) labelEl.textContent = "Dark";
  } else {
    if (iconEl) iconEl.textContent = "☀️";
    if (labelEl) labelEl.textContent = "Light";
  }
}

function toggleTheme() {
  const current =
    document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(current === "light" ? "dark" : "light");
}
applyTheme(localStorage.getItem("ragify_theme") || "light");

// --- Textarea Autosize & Key Navigation ---
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
}

// --- Chat Helpers ---
function scrollToBottom() {
  const chat = document.getElementById("chat-history");
  if (chat) chat.scrollTop = chat.scrollHeight;
}

function hideWelcomeScreen() {
  const el = document.getElementById("welcomeCenter");
  if (el) el.style.display = "none";
}

function renderUser(text) {
  hideWelcomeScreen();
  const chat = document.getElementById("chat-history");
  const div = document.createElement("div");
  div.className = "chat-bubble user";
  div.textContent = text;
  chat.appendChild(div);
  scrollToBottom();
}

function renderAssistantSlot() {
  hideWelcomeScreen();
  const chat = document.getElementById("chat-history");

  const wrapper = document.createElement("div");
  wrapper.className = "assistant-wrapper";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble assistant";

  const contentDiv = document.createElement("div");
  contentDiv.innerHTML = '<span class="cursor-blink"></span>';
  bubble.appendChild(contentDiv);

  const sourcesDiv = document.createElement("div");
  sourcesDiv.className = "sources-container";
  bubble.appendChild(sourcesDiv);

  wrapper.appendChild(bubble);

  const actionStrip = document.createElement("div");
  actionStrip.className = "assistant-actions";

  const copyBtn = document.createElement("button");
  copyBtn.className = "btn-action-ghost";
  copyBtn.setAttribute("data-tooltip", "Copy");
  copyBtn.innerHTML = ICON_COPY;
  copyBtn.onclick = () => copyText(copyBtn, contentDiv);
  actionStrip.appendChild(copyBtn);

  wrapper.appendChild(actionStrip);
  chat.appendChild(wrapper);
  scrollToBottom();

  return { contentDiv, sourcesDiv, bubble };
}

function formatMarkdownAndMath(element, text) {
  element.innerHTML = marked.parse(text);
  if (window.renderMathInElement) {
    renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}

function copyText(btn, contentEl) {
  const textToCopy = contentEl.innerText || contentEl.textContent;
  navigator.clipboard
    .writeText(textToCopy)
    .then(() => {
      btn.innerHTML = ICON_CHECK;
      btn.classList.add("copied");
      btn.setAttribute("data-tooltip", "Copied!");
      setTimeout(() => {
        btn.innerHTML = ICON_COPY;
        btn.classList.remove("copied");
        btn.setAttribute("data-tooltip", "Copy");
      }, 2000);
    })
    .catch((err) => {
      console.error("Copy failed", err);
    });
}

function persistTurn(question, answer, sources = []) {
  const history = JSON.parse(localStorage.getItem("ragify_history") || "[]");
  history.push({ question, answer, sources });
  localStorage.setItem("ragify_history", JSON.stringify(history));
}

function restoreHistory() {
  const history = JSON.parse(localStorage.getItem("ragify_history") || "[]");
  if (history.length) {
    hideWelcomeScreen();
    history.forEach((item) => {
      renderUser(item.question);
      const { contentDiv, sourcesDiv } = renderAssistantSlot();
      formatMarkdownAndMath(contentDiv, item.answer);
      if (item.sources && item.sources.length) {
        const unique = [...new Set(item.sources.map((s) => s.source || s))];
        sourcesDiv.innerHTML = unique
          .map((s) => `<span class="citation-tag">📄 ${s}</span>`)
          .join("");
      }
    });
    scrollToBottom();
  }
}

function resetConversation() {
  localStorage.removeItem("ragify_history");
  localStorage.removeItem("ragify_conv_id");
  conversationId =
    "conv_" +
    Math.random().toString(36).substring(2, 9) +
    Date.now().toString(36);
  localStorage.setItem("ragify_conv_id", conversationId);
  document.getElementById("chat-history").innerHTML = `
        <div class="welcome-center" id="welcomeCenter">
            <div class="welcome-icon">💬</div>
            <h2 class="welcome-title">How can I help with your documents?</h2>
            <p class="welcome-desc">Upload documents above and ask questions across all indexed content.</p>
        </div>
    `;
}

// --- Query Routing ---
async function submitQuery() {
  const input = document.getElementById("questionInput");
  const question = input.value.trim();
  if (!question) return;

  const streamToggle = document.getElementById("streamToggle");
  const isStream = streamToggle ? streamToggle.checked : true;

  if (isStream) {
    streamQuery(question);
  } else {
    standardQuery(question);
  }
}

async function standardQuery(question) {
  const input = document.getElementById("questionInput");
  const btn = document.getElementById("submitBtn");
  input.value = "";
  input.style.height = "auto";
  input.disabled = true;
  btn.disabled = true;

  renderUser(question);
  const { contentDiv, sourcesDiv } = renderAssistantSlot();

  try {
    const res = await fetch(
      `/ask?question=${encodeURIComponent(question)}&conversation_id=${encodeURIComponent(conversationId)}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
        credentials: "include",
      },
    );
    const data = await res.json();
    if (res.ok) {
      formatMarkdownAndMath(contentDiv, data.answer);
      if (data.sources && data.sources.length) {
        const unique = [...new Set(data.sources.map((s) => s.source || s))];
        sourcesDiv.innerHTML = unique
          .map((s) => `<span class="citation-tag">📄 ${s}</span>`)
          .join("");
      }
      persistTurn(question, data.answer, data.sources || []);
    } else {
      contentDiv.textContent = `Error: ${data.detail || "Query failed"}`;
    }
  } catch (err) {
    contentDiv.textContent = `Network error: ${err.message}`;
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
    scrollToBottom();
  }
}

async function streamQuery(question) {
  const input = document.getElementById("questionInput");
  const btn = document.getElementById("submitBtn");
  input.value = "";
  input.style.height = "auto";
  input.disabled = true;
  btn.disabled = true;

  renderUser(question);
  const { contentDiv } = renderAssistantSlot();

  try {
    const res = await fetch(
      `/ask/stream?question=${encodeURIComponent(question)}&conversation_id=${encodeURIComponent(conversationId)}`,
      {
        method: "POST",
        headers: getAuthHeaders(),
        credentials: "include",
      },
    );

    if (!res.ok) {
      const errText = await res.text();
      contentDiv.innerHTML = `<span style="color:#ef4444;">Error: ${errText || "Stream error"}</span>`;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      if (chunk) {
        fullText += chunk;
        contentDiv.textContent = fullText;

        const cursor = document.createElement("span");
        cursor.className = "cursor-blink";
        contentDiv.appendChild(cursor);

        scrollToBottom();
      }
    }

    formatMarkdownAndMath(contentDiv, fullText);
    persistTurn(question, fullText, []);
  } catch (err) {
    contentDiv.innerHTML = `<span style="color:#ef4444;">Stream failed: ${err.message}</span>`;
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
    scrollToBottom();
  }
}

// Boot lifecycle
window.addEventListener("DOMContentLoaded", async () => {
  await checkSession();
  restoreHistory();
});
