function apiFetch(url, options = {}) {
  return fetch(url, options).then(async r => {
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      throw new Error(err.detail || 'Request failed');
    }
    return r.json();
  });
}

const API = {
  async models() { return apiFetch('/models'); },
  async switchModel(m) { return apiFetch(`/models/switch?model=${encodeURIComponent(m)}`, {method:'POST'}); },
  async upload(file) {
    const fd = new FormData();
    fd.append('file', file);
    return apiFetch('/upload', {method:'POST', body:fd});
  },
  async uploadImage(file) {
    const fd = new FormData();
    fd.append('file', file);
    return apiFetch('/image/upload', {method:'POST', body:fd});
  },
  async editImage(req) {
    return apiFetch('/image/edit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
  },
  async stats() { return apiFetch('/memory/stats'); },
  async wipe() { return apiFetch('/memory/wipe', {method:'DELETE'}); },
  async getSessions() { return apiFetch('/sessions'); },
  async loadSession(id) { return apiFetch(`/sessions/${id}`, {method:'POST'}); },
  async newSession() { return apiFetch('/sessions/new', {method:'POST'}); },
  async clearContext() { return apiFetch('/memory/context', {method:'DELETE'}); },
  async getMemories() { return apiFetch('/memory/recent?limit=10'); },
  async draw(prompt, seed) {
    return apiFetch('/draw', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, seed: seed ? parseInt(seed) : null})
    });
  },
  async aiEditImage(req) {
    return apiFetch('/image/ai-edit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
  },
  async addModel(req) {
    return apiFetch('/models/add', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
  },
  async removeModel(modelId) {
    return apiFetch('/models/remove', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_id: modelId})
    });
  },
  async providers() { return apiFetch('/providers'); },
  async addProvider(req) {
    return apiFetch('/providers/add', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
  },
  async removeProvider(name) {
    return apiFetch(`/providers/remove?name=${encodeURIComponent(name)}`, {method:'POST'});
  },
  async ollamaCloudModels() { return apiFetch('/ollama-cloud-models'); },
  async pullModel(modelId) {
    return apiFetch(`/models/pull?model_id=${encodeURIComponent(modelId)}`, {method:'POST'});
  },
  async personalities() { return apiFetch('/personalities'); },
  async switchPersonality(id) {
    return apiFetch(`/personalities/switch?personality_id=${encodeURIComponent(id)}`, {method:'POST'});
  },
  async getWorkDir() { return apiFetch('/work-dir'); },
  async setWorkDir(path) {
    return apiFetch('/work-dir/set', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path})
    });
  },
  async gitUndo() { return apiFetch('/git/undo', {method:'POST'}); },
  async gitRedo() { return apiFetch('/git/redo', {method:'POST'}); },
  async getProjects() { return apiFetch('/projects'); },
  async switchProject(id) {
    return apiFetch('/projects/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({project_id: id})
    });
  },
  async removeProject(id) {
    return apiFetch(`/projects/${id}`, {method:'DELETE'});
  },
  async approveTool(reqId, approved) {
    return apiFetch(`/approve/${reqId}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({approved})
    });
  }
};

// ── State ────────────────────────────────────────────────────────────
let currentSessionId = null;
let isStreaming = false;
let chatHistory = [];
let currentEventSource = null;
let pendingAttachments = [];
let currentPersonalityId = 'standard';
let pendingApprovalReqId = null;

// ── DOM Elements ────────────────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const toggleSidebar = document.getElementById('toggle-sidebar');
const chatWindow = document.getElementById('chat-window');
const welcome = document.getElementById('welcome');
const msgInput = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');
const historyList = document.getElementById('history-list');
const toast = document.getElementById('toast');
const fileInput = document.getElementById('file-input');
const uploadBtnTool = document.getElementById('upload-btn-tool');
const uploadProgress = document.getElementById('upload-progress');
const uploadBar = document.getElementById('upload-bar');
const uploadResult = document.getElementById('upload-result');
const activeModelLabel = document.getElementById('active-model-label');

// ── Initialize ────────────────────────────────────────────────
async function init() {
  await loadModels();
  await refreshSessions();
  await refreshStats();
}
(async function startup() {
  autoResizeTextarea();
  await init();
  await loadPersonalities();
})();

async function loadPersonalities() {
  try {
    const data = await API.personalities();
    const container = document.getElementById('personality-selector');
    container.innerHTML = '';
    (data.personalities || []).forEach(p => {
      const item = document.createElement('div');
      item.className = `personality-item ${p.id === currentPersonalityId ? 'active' : ''}`;
      item.innerHTML = `
        <span class="personality-icon">${p.icon}</span>
        <div class="personality-info">
          <div class="personality-name">${p.name}</div>
          <div class="personality-desc">${p.description}</div>
        </div>
      `;
      item.dataset.id = p.id;
      item.onclick = () => switchPersonality(p.id);
      container.appendChild(item);
    });
  } catch(e) {
    console.error('Failed to load personalities:', e);
  }
}

async function switchPersonality(id) {
  if (id === currentPersonalityId) return;
  try {
    await API.switchPersonality(id);
    currentPersonalityId = id;
    document.querySelectorAll('.personality-item').forEach(el => el.classList.toggle('active', el.dataset.id === id));
    showToast(`Switched to ${id} personality`);
    updatePersonalitySidebarUI();
  } catch(e) {
    showToast('Failed to switch personality: ' + e.message);
  }
}

function updatePersonalitySidebarUI() {
  const container = document.getElementById('personality-sidebar-ui');
  if (currentPersonalityId === 'coding_agent') {
    container.innerHTML = `
<div class="sidebar-section">
  <div class="section-label">Recent Projects</div>
  <div id="project-list" style="padding:0 12px 4px;"></div>
  <div class="section-label">Working Directory</div>
  <div style="padding:4px 12px 12px;">
    <input type="text" id="work-dir-input" placeholder="C:\\Path\\To\\Project" style="width:100%;padding:8px;border-radius:6px;background:#1e1e1e;color:#fff;border:1px solid #333;font-size:12px;box-sizing:border-box;" />
    <button id="set-work-dir-btn" class="tool-btn" style="justify-content:center;margin-top:6px;padding:6px;">Set Directory</button>
    <div id="work-dir-status" style="font-size:11px;color:#888;margin-top:4px;"></div>
  </div>
  <div style="display:flex;gap:6px;padding:0 12px 12px;">
    <button id="git-undo-btn" class="tool-btn" style="flex:1;justify-content:center;padding:6px;font-size:12px;" title="Undo last commit">↩ Undo</button>
    <button id="git-redo-btn" class="tool-btn" style="flex:1;justify-content:center;padding:6px;font-size:12px;" title="Redo undone commit">↪ Redo</button>
  </div>
</div>`;
    container.style.display = 'block';
    refreshProjects();
    setTimeout(() => {
      const wdInput = document.getElementById('work-dir-input');
      const setBtn = document.getElementById('set-work-dir-btn');
      const status = document.getElementById('work-dir-status');
      if (setBtn) {
        setBtn.onclick = async () => {
          const path = wdInput.value.trim();
          if (!path) { status.textContent = 'Please enter a path'; return; }
          setBtn.textContent = 'Setting...';
          setBtn.disabled = true;
          try {
            const data = await API.setWorkDir(path);
            const isErr = data.message.includes('[ERROR]');
            status.textContent = data.message.replace('[SUCCESS] ', '').replace('[ERROR] ', '');
            status.style.color = isErr ? '#ff5b5b' : '#81c995';
            if (!isErr) wdInput.value = path;
            refreshProjects();
          } catch(e) {
            status.textContent = 'Error: ' + e.message;
            status.style.color = '#ff5b5b';
          }
          setBtn.textContent = 'Set Directory';
          setBtn.disabled = false;
        };
        wdInput.onkeydown = (e) => { if (e.key === 'Enter') setBtn.click(); };
      }
      // Load current work_dir into input
      API.getWorkDir().then(d => {
        if (wdInput && d.work_dir) wdInput.value = d.work_dir;
      }).catch(() => {});
      const undoBtn = document.getElementById('git-undo-btn');
      const redoBtn = document.getElementById('git-redo-btn');
      if (undoBtn) undoBtn.onclick = async () => {
        undoBtn.textContent = 'Undoing...';
        undoBtn.disabled = true;
        try {
          const data = await API.gitUndo();
          showToast(data.message || 'Undo completed');
        } catch(e) { showToast('Undo failed: ' + e.message); }
        undoBtn.textContent = '↩ Undo';
        undoBtn.disabled = false;
      };
      if (redoBtn) redoBtn.onclick = async () => {
        redoBtn.textContent = 'Redoing...';
        redoBtn.disabled = true;
        try {
          const data = await API.gitRedo();
          showToast(data.message || 'Redo completed');
        } catch(e) { showToast('Redo failed: ' + e.message); }
        redoBtn.textContent = '↪ Redo';
        redoBtn.disabled = false;
      };
    }, 0);
  } else {
    container.style.display = 'none';
    container.innerHTML = '';
  }
}

async function refreshProjects() {
  const list = document.getElementById('project-list');
  if (!list) return;
  try {
    const [projData, wdData] = await Promise.all([API.getProjects(), API.getWorkDir()]);
    const currentDir = (wdData.work_dir || '').toLowerCase().replace(/\\/g, '/');
    const projects = projData.projects || [];
    if (projects.length === 0) {
      list.innerHTML = '<div style="padding:4px 0 8px;font-size:12px;color:var(--text-muted);">No recent projects</div>';
      return;
    }
    list.innerHTML = projects.map(p => {
      const pPath = (p.path || '').toLowerCase().replace(/\\/g, '/');
      const isActive = currentDir && pPath === currentDir;
      return `
      <div class="project-item ${isActive ? 'project-active' : ''}" data-id="${p.id}" data-path="${p.path}" title="${p.path}">
        <span class="project-icon">${isActive ? '📂' : '📁'}</span>
        <span class="project-name">${p.name}</span>
        <span class="project-date">${new Date(p.timestamp * 1000).toLocaleDateString()}</span>
        <button class="project-delete" title="Remove from recent">✕</button>
      </div>`;
    }).join('');
    list.querySelectorAll('.project-item').forEach(el => {
      el.addEventListener('click', async (e) => {
        if (e.target.classList.contains('project-delete')) return;
        const pid = el.dataset.id;
        const path = el.dataset.path;
        try {
          const res = await API.switchProject(pid);
          const wdInput = document.getElementById('work-dir-input');
          if (wdInput) wdInput.value = path;
          const wdStatus = document.getElementById('work-dir-status');
          if (wdStatus) { wdStatus.textContent = 'Active: ' + path; wdStatus.style.color = '#81c995'; }
          refreshProjects();
        } catch(err) {
          showToast('Failed to switch project: ' + err.message);
        }
      });
      el.querySelector('.project-delete').addEventListener('click', async (e) => {
        e.stopPropagation();
        const pid = el.dataset.id;
        try {
          await API.removeProject(pid);
          refreshProjects();
        } catch(err) {
          showToast('Failed to remove project');
        }
      });
    });
  } catch(e) {
    list.innerHTML = '<div style="padding:4px 0 8px;font-size:12px;color:var(--text-muted);">Failed to load projects</div>';
  }
}

// ── Functions ────────────────────────────────────────────────────
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3000);
}

function autoResizeTextarea() {
  msgInput.addEventListener('input', () => {
    msgInput.style.height = 'auto';
    msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + 'px';
  });
}

function renderPendingAttachments() {
  const container = document.getElementById('pending-attachments');
  container.innerHTML = '';
  if (pendingAttachments.length === 0) {
    container.style.display = 'none';
    return;
  }
  container.style.display = 'flex';
  pendingAttachments.forEach((att, idx) => {
    const chip = document.createElement('div');
    chip.style.cssText = 'display:flex;align-items:center;gap:6px;padding:4px 8px;background:var(--surface);border:1px solid var(--border);border-radius:12px;font-size:12px;color:var(--text-secondary);cursor:default;';
    const info = att.pdf_info || {};
    const meta = [];
    if (info.page_count) meta.push(`${info.page_count}p`);
    if (info.file_size_display) meta.push(info.file_size_display);
    const metaStr = meta.length ? ` (${meta.join(', ')})` : '';
    chip.innerHTML = `
      <span>📄 ${att.displayName || att.filename}${metaStr}</span>
      <span style="cursor:pointer;color:var(--danger);font-weight:bold;" onclick="removePendingAttachment(${idx})">&times;</span>
    `;
    container.appendChild(chip);
  });
}

function removePendingAttachment(idx) {
  pendingAttachments.splice(idx, 1);
  renderPendingAttachments();
}

async function loadModels() {
  try {
    const data = await API.models();
    activeModelLabel.textContent = data.active || '—';
    console.log('Models loaded:', data.models?.length || 0);
  } catch(e) {
    console.error('Model load error:', e);
    activeModelLabel.textContent = 'Error';
    showToast('Could not load models: ' + (e.name === 'AbortError' ? 'timeout' : e.message));
  }
}

async function refreshSessions() {
  try {
    const data = await API.getSessions();
    currentSessionId = data.active_id || null;
    historyList.innerHTML = '';
    
    if (!data.sessions || data.sessions.length === 0) {
      historyList.innerHTML = '<div style="padding:12px; color:var(--text-muted); font-size:12px;">No chats yet</div>';
      return;
    }
    
    data.sessions.forEach(sess => {
      const btn = document.createElement('button');
      btn.className = `history-item ${sess.id === currentSessionId ? 'active' : ''}`;
      btn.innerHTML = `
        <span class="history-item-icon">💬</span>
        <span class="history-item-title">${sess.title}</span>
        <span class="history-item-date">${new Date(sess.timestamp * 1000).toLocaleDateString()}</span>
      `;
      btn.onclick = () => switchSession(sess.id);
      historyList.appendChild(btn);
    });
  } catch(e) {
    console.error('Failed to refresh sessions:', e);
  }
};

async function refreshStats() {
  try {
    const s = await API.stats();
    const el = document.getElementById('memory-count');
    if (el) el.textContent = s.total_records || '0';
  } catch(e) {}
}

// ── Markdown Rendering ──────────────────────────────────────
function renderMarkdown(bubble) {
  const raw = bubble.getAttribute('data-raw') || bubble.textContent;
  bubble.setAttribute('data-raw', raw);

  let html;
  try {
    html = marked.parse(raw, { breaks: true, gfm: true });
  } catch(e) {
    html = `<p>${raw.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</p>`;
  }

  bubble.innerHTML = html;
  bubble.querySelectorAll('pre code').forEach((block) => {
    const lang = block.className.replace(/^lang-/, '') || 'code';
    const code = block.textContent;

    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';

    const header = document.createElement('div');
    header.className = 'code-header';
    const langLabel = document.createElement('span');
    langLabel.className = 'code-lang';
    langLabel.textContent = lang;
    const copyBtn = document.createElement('button');
    copyBtn.className = 'code-copy-btn';
    copyBtn.textContent = 'Copy';
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(code).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = 'Copy', 2000);
      });
    };
    header.appendChild(langLabel);
    header.appendChild(copyBtn);

    const pre = document.createElement('pre');
    const codeEl = document.createElement('code');
    codeEl.className = block.className;
    codeEl.textContent = code;
    pre.appendChild(codeEl);

    wrapper.appendChild(header);
    wrapper.appendChild(pre);
    block.parentNode.replaceChild(wrapper, block);
  });

  try { hljs.highlightAll(); } catch(e) {}
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Chat Functions ────────────────────────────────────────────
function renderCodeBlocks(bubble) {
  renderMarkdown(bubble);
}

function renderFilePreviewCard(att) {
  const fileName = att.filename || att;
  const displayName = att.displayName || fileName;
  const fileType = att.type || '';
  const info = att.pdf_info || {};
  const thumbUrl = att.thumbnail_url || null;
  const pageCount = info.page_count || 0;

  const card = document.createElement('div');
  card.className = 'pdf-preview-card';

  let icon = '📄', iconHtml = '';
  let metaParts = [];
  let openLabel = 'Open';

  if (fileType === 'pdf' || fileName.toLowerCase().endsWith('.pdf')) {
    icon = '📄';
    openLabel = 'Open PDF';
    if (thumbUrl) {
      iconHtml = `<img class="pdf-thumb" src="${thumbUrl}" alt="PDF preview" loading="lazy"/>`;
    } else {
      iconHtml = `<div class="pdf-thumb pdf-thumb-placeholder">📄</div>`;
    }
    if (pageCount > 0) metaParts.push(`${pageCount} page${pageCount > 1 ? 's' : ''}`);
    metaParts.push('Ingested');
  } else if (fileType === 'image' || fileName.match(/\.(png|jpg|jpeg|gif|webp)$/i)) {
    icon = '🖼️';
    openLabel = 'View Image';
    if (thumbUrl && thumbUrl.startsWith('/uploads/')) {
      iconHtml = `<img class="pdf-thumb" src="${thumbUrl}" alt="Image preview" style="object-fit:cover;" loading="lazy"/>`;
    } else {
      iconHtml = `<div class="pdf-thumb pdf-thumb-placeholder">🖼️</div>`;
    }
    metaParts.push('Image');
  } else if (fileType === 'text' || fileName.match(/\.(txt|py|js|ts|html|css|json|md|csv|xml|yaml|yml|sh|bat|ps1)$/i)) {
    icon = '📝';
    openLabel = 'View File';
    iconHtml = `<div class="pdf-thumb pdf-thumb-placeholder">📝</div>`;
    metaParts.push('Text');
  } else {
    iconHtml = `<div class="pdf-thumb pdf-thumb-placeholder">📎</div>`;
    metaParts.push('File');
  }

  card.innerHTML = `
    ${iconHtml}
    <div class="pdf-card-body">
      <div class="pdf-card-name">${displayName}</div>
      <div class="pdf-card-meta">
        <span>${metaParts.join(' · ')}</span>
      </div>
      <a href="/uploads/${fileName}" target="_blank" class="pdf-open-btn" title="${openLabel}">📂 ${openLabel}</a>
    </div>
  `;
  return card;
}

function addMessage(role, text, attachments = []) {
  const bubble = document.createElement('div');
  const displayRole = role === 'assistant' ? 'bot' : role;
  bubble.className = `msg ${displayRole}`;
  
  if (displayRole === 'bot') {
    bubble.setAttribute('data-raw', text);
    renderCodeBlocks(bubble);
  } else {
    if (attachments && attachments.length > 0) {
      const attContainer = document.createElement('div');
      attContainer.className = 'attachments-container';
      attachments.forEach(att => {
        const card = renderFilePreviewCard(att);
        attContainer.appendChild(card);
      });
      bubble.appendChild(attContainer);
    }
    
    const textSpan = document.createElement('span');
    textSpan.className = 'msg-text';
    textSpan.textContent = text;
    bubble.appendChild(textSpan);
  }
  
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

function addAttachmentMessage(att) {
  const bubble = document.createElement('div');
  bubble.className = 'msg user attachment';
  const card = renderFilePreviewCard(att);
  bubble.appendChild(card);
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addAttachmentMessage(att) {
  const bubble = document.createElement('div');
  bubble.className = 'msg user attachment';
  const fileName = att.filename || att;
  const displayName = att.displayName || fileName;
  bubble.innerHTML = `
    <div class="attachment-container">
      <span class="attachment-icon">📄</span>
      <a href="/uploads/${fileName}" target="_blank" class="attachment-name" style="color:var(--accent); text-decoration:none;">${displayName}</a>
      <span class="attachment-status">Uploaded</span>
    </div>
  `;
  chatWindow.appendChild(bubble);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if ((!text && pendingAttachments.length === 0) || isStreaming) return;
  
  msgInput.value = '';
  isStreaming = true;
  sendBtn.disabled = true;
  document.getElementById('stop-btn').style.display = 'inline-block';
  
  // Add user message with any pending attachments
  addMessage('user', text, pendingAttachments);
  
  const botBubble = addMessage('bot', '');
  botBubble.setAttribute('data-raw', '');
  botBubble.classList.add('streaming');
  welcome.classList.add('hidden');
  
  // Clear pending attachments
  const currentAttachments = [...pendingAttachments];
  pendingAttachments = [];
  renderPendingAttachments();
  
  try {
    const params = new URLSearchParams({ message: text });
    if (currentSessionId) params.set('session_id', currentSessionId);
    
    // Send attachments as a comma-separated list
    if (currentAttachments.length > 0) {
      params.set('attachments', currentAttachments.map(a => a.displayName || a.filename).join(','));
    }
    
    const source = new EventSource(`/chat?${params}`);
    currentEventSource = source;
    source.onmessage = (e) => {
      if (e.data === '[DONE]') {
        source.close();
        isStreaming = false;
        sendBtn.disabled = false;
        document.getElementById('stop-btn').style.display = 'none';
        currentEventSource = null;
        botBubble.classList.remove('streaming');
        renderCodeBlocks(botBubble);
        refreshSessions();
        return;
      }
      try {
        const d = JSON.parse(e.data);
        if (d.token) {
          const raw = botBubble.getAttribute('data-raw') + d.token;
          botBubble.setAttribute('data-raw', raw);
          botBubble.textContent = raw;
        } else if (d.type === 'approval') {
          showApprovalDialog(d.req_id, d.tool, d.args);
        } else if (d.type === 'approval_result') {
          hideApprovalDialog();
          if (d.approved) {
            showToast('Tool approved and executed');
          } else {
            showToast('Tool execution denied');
          }
        }
      } catch(_) {}
    };
    source.onerror = () => {
      source.close();
      isStreaming = false;
      sendBtn.disabled = false;
      document.getElementById('stop-btn').style.display = 'none';
      currentEventSource = null;
      botBubble.classList.remove('streaming');
      if (!botBubble.getAttribute('data-raw')) {
        botBubble.textContent = 'Error: Failed to get response';
      }
      renderCodeBlocks(botBubble);
      refreshSessions();
    };
  } catch(e) {
    isStreaming = false;
    sendBtn.disabled = false;
    document.getElementById('stop-btn').style.display = 'none';
    currentEventSource = null;
    botBubble.classList.remove('streaming');
    botBubble.textContent = 'Error: ' + e.message;
    refreshSessions();
  }
}

async function createNewChat() {
  try {
    const data = await API.newSession();
    currentSessionId = data.id;
    chatWindow.innerHTML = '';
    welcome.classList.remove('hidden');
    chatHistory = [];
    refreshSessions();
  } catch(e) {
    showToast('Failed to create new chat');
  }
}

async function switchSession(id) {
  try {
    const data = await API.loadSession(id);
    currentSessionId = id;
    chatWindow.innerHTML = '';
    welcome.classList.add('hidden');
    if (data.messages && data.messages.length > 0) {
      data.messages.forEach(m => {
        addMessage(m.role || 'user', m.content || m);
      });
    }
    if (data.attachments && data.attachments.length > 0) {
      data.attachments.forEach(att => {
        addAttachmentMessage(att);
      });
    }
    refreshSessions();
  } catch(e) {
    showToast('Failed to load session');
  }
}

// ── Event Listeners ────────────────────────────────────────────
toggleSidebar.onclick = () => {
  document.body.classList.toggle('sidebar-collapsed');
};

document.getElementById('new-chat-btn').onclick = createNewChat;

sendBtn.onclick = sendMessage;
msgInput.onkeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
};

function createLoadingSpinner(text) {
  const el = document.createElement('div');
  el.className = 'gen-loading';
  el.innerHTML = `
    <div class="gen-spinner"><div class="spinner-ring"></div></div>
    <div class="gen-status">${text}</div>
  `;
  return el;
}

document.getElementById('draw-btn').onclick = async () => {
  const prompt = msgInput.value.trim();
  if (!prompt) {
    showToast('Enter a prompt for image generation');
    return;
  }
  
  addMessage('user', `🎨 ${prompt}`);
  msgInput.value = '';
  
  const botBubble = addMessage('bot', '');
  const spinner = createLoadingSpinner('Initializing model...');
  botBubble.appendChild(spinner);
  botBubble.classList.add('streaming');
  
  const statusEl = spinner.querySelector('.gen-status');
  const phases = [
    { after: 2000, text: 'Loading Stable Diffusion...' },
    { after: 5000, text: 'Generating image...' },
    { after: 15000, text: 'Still working... almost there!' },
  ];
  const timers = phases.map(p => setTimeout(() => { if (statusEl) statusEl.textContent = p.text; }, p.after));
  
  try {
    const data = await API.draw(prompt);
    timers.forEach(clearTimeout);
    botBubble.classList.remove('streaming');
    botBubble.innerHTML = '';
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg-text';
    msgDiv.textContent = data.message;
    botBubble.appendChild(msgDiv);
    if (data.filename) {
      const img = document.createElement('img');
      img.src = `/images/${data.filename}`;
      img.style.maxWidth = '100%';
      img.style.borderRadius = '8px';
      img.style.marginTop = '12px';
      botBubble.appendChild(img);
    }
  } catch(e) {
    timers.forEach(clearTimeout);
    botBubble.classList.remove('streaming');
    botBubble.innerHTML = '<div class="msg-text">Error generating image</div>';
  }
};

document.getElementById('clear-btn').onclick = async () => {
  await createNewChat();
  showToast('Started new chat');
};

document.getElementById('upload-btn').onclick = () => fileInput.click();
fileInput.onchange = async () => {
  if (!fileInput.files[0]) return;
  
  const file = fileInput.files[0];
  const allowedExts = ['.pdf','.png','.jpg','.jpeg','.gif','.webp','.txt','.py','.js','.ts','.html','.css','.json','.md','.csv','.xml','.yaml','.yml','.sh','.bat','.ps1'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowedExts.includes(ext)) {
    showToast('File type not supported');
    return;
  }
  
  uploadProgress.classList.add('active');
  uploadBar.style.width = '30%';
  
  try {
    const r = await API.upload(file);
    uploadBar.style.width = '100%';
    if (r.success) {
      uploadResult.className = 'upload-result success';
      uploadResult.textContent = r.message;
      pendingAttachments.push({ filename: r.attachment.filename, displayName: file.name, info: r.attachment });
      renderPendingAttachments();
    } else {
      uploadResult.className = 'upload-result error';
      uploadResult.textContent = r.message;
    }
    refreshStats();
  } catch(e) {
    uploadResult.className = 'upload-result error';
    uploadResult.textContent = 'Upload failed';
  }
  
  setTimeout(() => {
    uploadProgress.classList.remove('active');
    uploadBar.style.width = '0%';
  }, 2000);
};

// Hide the upload button in sidebar tools since it's now in main area
document.getElementById('upload-btn-tool').style.display = 'none';

document.getElementById('wipe-btn').onclick = async () => {
  if (!confirm('Wipe all memory? This cannot be undone.')) return;
  
  try {
    await API.wipe();
    chatWindow.innerHTML = '';
    welcome.classList.remove('hidden');
    showToast('Memory wiped');
    refreshStats();
  } catch(e) {
    showToast('Failed to wipe memory');
  }
};

document.getElementById('view-mem-btn').onclick = async () => {
  try {
    const data = await API.getMemories();
    if (!data.memories || data.memories.length === 0) {
      showToast('No memories found');
      return;
    }
    
    let text = '🧠 Recent Memories:\n\n';
    data.memories.forEach(m => {
      text += `[${m.role}] ${m.content.substring(0, 100)}...\n\n`;
    });
    
    addMessage('bot', text);
  } catch(e) {
    showToast('Failed to load memories');
  }
};

document.getElementById('stop-btn').onclick = () => {
  if (currentEventSource) {
    currentEventSource.close();
    isStreaming = false;
    sendBtn.disabled = false;
    document.getElementById('stop-btn').style.display = 'none';
    currentEventSource = null;
    if (currentSessionId) {
      fetch(`/cancel?session_id=${encodeURIComponent(currentSessionId)}`, { method: 'POST' }).catch(() => {});
    }
    showToast('Response stopped');
  }
};

// ── Approval Dialog ──────────────────────────────────────────────
function showApprovalDialog(reqId, toolName, args) {
  pendingApprovalReqId = reqId;
  document.getElementById('approval-tool-name').textContent = toolName;
  document.getElementById('approval-tool-args').textContent = JSON.stringify(args, null, 2);
  document.getElementById('approval-dialog').style.display = 'flex';
}

function hideApprovalDialog() {
  pendingApprovalReqId = null;
  document.getElementById('approval-dialog').style.display = 'none';
}

document.getElementById('approval-approve-btn').onclick = async () => {
  if (!pendingApprovalReqId) return;
  const reqId = pendingApprovalReqId;
  hideApprovalDialog();
  try {
    await API.approveTool(reqId, true);
  } catch(e) {
    showToast('Failed to send approval: ' + e.message);
  }
};

document.getElementById('approval-deny-btn').onclick = async () => {
  if (!pendingApprovalReqId) return;
  const reqId = pendingApprovalReqId;
  hideApprovalDialog();
  try {
    await API.approveTool(reqId, false);
  } catch(e) {
    showToast('Failed to send denial: ' + e.message);
  }
};

// ── Model & Provider Modals ─────────────────────────────────────

function showModelModal() {
  document.getElementById('model-modal').style.display = 'flex';
}

function closeModelModal() {
  document.getElementById('model-modal').style.display = 'none';
}

document.getElementById('model-modal-close').onclick = closeModelModal;
document.getElementById('model-modal').onclick = (e) => {
  if (e.target === e.currentTarget) closeModelModal();
};

function populateModelList(models, current, cloud, local, ollamaCloud = []) {
  const list = document.getElementById('model-modal-list');
  const currentLabel = document.getElementById('model-modal-current');
  currentLabel.textContent = current || '—';
  list.innerHTML = '';

  if (!models && !cloud && !local && !ollamaCloud) {
    const empty = document.createElement('div');
    empty.style.padding = '20px';
    empty.style.textAlign = 'center';
    empty.style.color = '#666';
    empty.textContent = 'No models found';
    list.appendChild(empty);
    return;
  }

  function addSection(title, items, isOllamaCloud = false) {
    if (!items || items.length === 0) return;
    const label = document.createElement('div');
    label.style.cssText = 'font-size:11px;color:#666;padding:8px 4px 4px;text-transform:uppercase;letter-spacing:0.5px;';
    label.textContent = title;
    list.appendChild(label);
    items.forEach(m => {
      const item = document.createElement('div');
      item.className = `model-item ${m === current ? 'active' : ''}`;
      
      if (isOllamaCloud) {
        // For Ollama Cloud models, m is an object with name, description, etc.
        const providerIcon = '☁️';
        const displayName = m.name || 'Unknown Model';
        const provider = 'ollama-cloud';
        const tags = (m.tags || []).slice(0, 3).join(' · '); // Show first 3 tags
        
        item.innerHTML = `
          <div class="model-check">${m === current ? '✓' : ''}</div>
          <span class="model-name">${providerIcon} ${displayName}</span>
          ${tags ? `<span class="model-tags" style="font-size:10px;color:#888;margin-left:8px;">${tags}</span>` : ''}
          <span class="model-size" style="font-size:10px;color:#888;">${provider}</span>
        `;
        
        // Add pull button for Ollama Cloud models
        if (!(m === current)) {
          const pullBtn = document.createElement('button');
          pullBtn.className = 'model-pull-btn';
          pullBtn.textContent = 'Pull';
          pullBtn.onclick = async () => {
            pullBtn.textContent = 'Pulling...';
            pullBtn.disabled = true;
            try {
              await API.pullModel(m.name);
              showToast(`Started pulling ${m.name}`);
              // Refresh model list after a delay to allow pull to start
              setTimeout(() => {
                API.models().then(data => {
                  populateModelList(data.models || [], data.active, data.cloud || [], data.local || [], data.ollamaCloud || []);
                  activeModelLabel.textContent = data.active || '—';
                });
              }, 2000);
            } catch(e) {
              pullBtn.textContent = 'Pull';
              pullBtn.disabled = false;
              showToast('Failed to start pull: ' + e.message);
            }
          };
          item.appendChild(pullBtn);
        }
      } else {
        const isCloud = typeof m === 'string' && m.includes(':');
        const displayName = isCloud ? m.split(':')[1] : m;
        const provider = isCloud ? m.split(':')[0] : 'ollama';
        const providerIcon = provider === 'openai' ? '🔷' : provider === 'anthropic' ? '🟣' : provider === 'gemini' ? '✨' : isCloud ? '🔌' : '💻';
        item.innerHTML = `
          <div class="model-check">${m === current ? '✓' : ''}</div>
          <span class="model-name">${providerIcon} ${displayName}</span>
          <span class="model-size" style="font-size:10px;color:#888;">${provider}</span>
        `;
      }
      
      item.onclick = async () => {
        if (isOllamaCloud) {
          if (m === current) { closeModelModal(); return; }
          try {
            // For Ollama Cloud models, we need to pull first then switch
            await API.switchModel(`ollama-cloud:${m.name}`);
            activeModelLabel.textContent = `ollama-cloud:${m.name}`;
            showToast(`Switched to ${m.name}`);
            populateModelList(models, `ollama-cloud:${m.name}`, cloud, local, ollamaCloud);
          } catch(e) {
            showToast('Failed to switch model: ' + e.message);
          }
        } else {
          if (m === current) { closeModelModal(); return; }
          try {
            await API.switchModel(m);
            activeModelLabel.textContent = m;
            showToast(`Switched to ${m}`);
            populateModelList(models, m, cloud, local, ollamaCloud);
          } catch(e) {
            showToast('Failed to switch model: ' + e.message);
          }
        }
      };
      
      list.appendChild(item);
    });
  }

  addSection('Local (Ollama)', local);
  addSection('Cloud', cloud);
  addSection('☁️ Ollama Cloud', ollamaCloud, true);
}

async function refreshProviderSelect() {
  const sel = document.getElementById('add-model-provider');
  const current = sel.value;
  sel.innerHTML = '<option value="ollama">Ollama (Local)</option>';
  try {
    const data = await API.providers();
    (data.providers || []).forEach(p => {
      if (p.name !== 'ollama') {
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.name.charAt(0).toUpperCase() + p.name.slice(1) + (p.base_url ? '' : '');
        sel.appendChild(opt);
      }
    });
  } catch(e) {}
  if ([...sel.options].some(o => o.value === current)) sel.value = current;
}

function showAddModelModal() {
  refreshProviderSelect();
  document.getElementById('add-model-modal').style.display = 'flex';
  document.getElementById('add-model-error').textContent = '';
  document.getElementById('add-model-id').value = '';
  document.getElementById('add-model-api-key').value = '';
  toggleApiKeyField();
}

function closeAddModelModal() {
  document.getElementById('add-model-modal').style.display = 'none';
}

function toggleApiKeyField() {
  const provider = document.getElementById('add-model-provider').value;
  const section = document.getElementById('add-model-api-section');
  section.style.display = provider === 'ollama' ? 'none' : 'block';
  if (provider !== 'ollama') {
    document.getElementById('add-model-api-key').placeholder = 'Leave blank to reuse stored key';
  }
}

document.getElementById('add-model-btn').onclick = showAddModelModal;
document.getElementById('add-model-provider').onchange = toggleApiKeyField;
document.getElementById('add-model-close').onclick = closeAddModelModal;
document.getElementById('add-model-modal').onclick = (e) => {
  if (e.target === e.currentTarget) closeAddModelModal();
};
document.getElementById('add-model-add-provider-link').onclick = (e) => {
  e.preventDefault();
  closeAddModelModal();
  showProvidersModal();
};

document.getElementById('add-model-submit').onclick = async () => {
  const provider = document.getElementById('add-model-provider').value;
  const modelId = document.getElementById('add-model-id').value.trim();
  const apiKey = document.getElementById('add-model-api-key').value.trim();
  const errEl = document.getElementById('add-model-error');

  if (!modelId) {
    errEl.textContent = 'Please enter a model ID';
    return;
  }

  try {
    const btn = document.getElementById('add-model-submit');
    btn.textContent = provider === 'ollama' ? 'Pulling model...' : 'Adding model...';
    btn.disabled = true;
    await API.addModel({ model_id: modelId, provider, api_key: apiKey || undefined });
    showToast(`Model ${modelId} added!`);
    closeAddModelModal();
    const data = await API.models();
    populateModelList(data.models || [], data.active, data.cloud || [], data.local || []);
    activeModelLabel.textContent = data.active || '—';
  } catch(e) {
    errEl.textContent = e.message;
  } finally {
    const btn = document.getElementById('add-model-submit');
    btn.textContent = 'Add Model';
    btn.disabled = false;
  }
};

document.getElementById('model-btn').onclick = async () => {
  try {
    const data = await API.models();
    populateModelList(data.models || [], data.active, data.cloud || [], data.local || []);
    if (!data.models || data.models.length === 0) {
      showToast('No models found. Add one via the model selector.');
      return;
    }
    showModelModal();
  } catch(e) {
    console.error(e);
    showToast('Error: ' + (e.name === 'AbortError' ? 'timeout (Ollama not running?)' : e.message));
  }
};

// ── Provider Management ────────────────────────────────────────

function showProvidersModal() {
  refreshProvidersList();
  document.getElementById('providers-modal').style.display = 'flex';
}

function closeProvidersModal() {
  document.getElementById('providers-modal').style.display = 'none';
}

document.getElementById('providers-modal-close').onclick = closeProvidersModal;
document.getElementById('providers-modal').onclick = (e) => {
  if (e.target === e.currentTarget) closeProvidersModal();
};

async function refreshProvidersList() {
  const list = document.getElementById('providers-list');
  list.innerHTML = '<div style="color:#888;text-align:center;padding:20px;">Loading...</div>';
  try {
    const data = await API.providers();
    list.innerHTML = '';
    (data.providers || []).forEach(p => {
      const card = document.createElement('div');
      card.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-radius:8px;background:#1e1e1e;border:1px solid #333;margin-bottom:6px;';
      const isBuiltin = ['ollama', 'openai', 'anthropic'].includes(p.name);
      const icon = p.name === 'ollama' ? '💻' : p.name === 'openai' ? '🔷' : p.name === 'anthropic' ? '🟣' : '🔌';
      card.innerHTML = `
        <div>
          <div style="font-size:14px;color:#ccc;">${icon} ${p.name} <span style="font-size:11px;color:#888;">${p.type}</span></div>
          <div style="font-size:11px;color:#666;margin-top:2px;">${p.models.length} model(s)${p.base_url ? ' · ' + p.base_url : ''}</div>
        </div>
        ${isBuiltin ? '<span style="font-size:11px;color:#666;">built-in</span>' : '<button class="remove-provider-btn" style="background:#ff5b5b33;border:1px solid #ff5b5b;color:#ff5b5b;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;">Remove</button>'}
      `;
      if (!isBuiltin) {
        card.querySelector('.remove-provider-btn').onclick = async () => {
          if (!confirm(`Remove provider "${p.name}" and all its models?`)) return;
          try {
            await API.removeProvider(p.name);
            showToast(`Provider "${p.name}" removed`);
            refreshProvidersList();
          } catch(e) {
            showToast('Failed to remove provider: ' + e.message);
          }
        };
      }
      list.appendChild(card);
    });
  } catch(e) {
    list.innerHTML = '<div style="color:#ff5b5b;text-align:center;padding:20px;">Failed to load providers</div>';
  }
}

document.getElementById('providers-btn').onclick = showProvidersModal;
document.getElementById('add-provider-btn').onclick = () => {
  closeProvidersModal();
  showAddProviderModal();
};

// ── Add Provider Modal ─────────────────────────────────────────

function toggleProviderFields() {
  const type = document.getElementById('add-provider-type').value;
  const baseUrlSection = document.getElementById('add-provider-base-url-section');
  baseUrlSection.style.display = type === 'openai_compatible' ? 'block' : 'none';
}

function showAddProviderModal() {
  document.getElementById('add-provider-modal').style.display = 'flex';
  document.getElementById('add-provider-error').textContent = '';
  document.getElementById('add-provider-name').value = '';
  document.getElementById('add-provider-base-url').value = '';
  document.getElementById('add-provider-api-key').value = '';
  toggleProviderFields();
}

function closeAddProviderModal() {
  document.getElementById('add-provider-modal').style.display = 'none';
}

document.getElementById('add-provider-close').onclick = closeAddProviderModal;
document.getElementById('add-provider-modal').onclick = (e) => {
  if (e.target === e.currentTarget) closeAddProviderModal();
};

document.getElementById('add-provider-submit').onclick = async () => {
  const name = document.getElementById('add-provider-name').value.trim();
  const type = document.getElementById('add-provider-type').value;
  const baseUrl = document.getElementById('add-provider-base-url').value.trim();
  const apiKey = document.getElementById('add-provider-api-key').value.trim();
  const errEl = document.getElementById('add-provider-error');

  if (!name) { errEl.textContent = 'Provider name is required'; return; }
  if (type === 'openai_compatible' && !baseUrl) { errEl.textContent = 'Base URL is required for OpenAI Compatible'; return; }
  if (!apiKey) { errEl.textContent = 'API key is required'; return; }

  try {
    const btn = document.getElementById('add-provider-submit');
    btn.textContent = 'Adding & detecting models...';
    btn.disabled = true;
    const data = await API.addProvider({ name, type, api_key: apiKey, base_url: baseUrl || undefined });
    const count = (data.detected_models || []).length;
    showToast(`Provider "${name}" added! ${count} model(s) auto-detected.`);
    closeAddProviderModal();
  } catch(e) {
    errEl.textContent = e.message;
  } finally {
    const btn = document.getElementById('add-provider-submit');
    btn.textContent = 'Add Provider';
    btn.disabled = false;
  }
};

// ── Image Editor ─────────────────────────────────────────────────────
let currentImagePath = null;

function openImageEditor() {
  document.getElementById('img-editor-modal').style.display = 'flex';
  currentImagePath = null;
  document.getElementById('img-preview').style.display = 'none';
  document.getElementById('edit-actions').style.display = 'none';
  document.getElementById('edit-result').style.display = 'none';
  document.getElementById('img-upload-area').style.display = 'block';
}

function closeImageEditor() {
  document.getElementById('img-editor-modal').style.display = 'none';
}

function updateEditParams() {
  const action = document.getElementById('edit-action-select').value;
  const paramsDiv = document.getElementById('edit-params');
  let html = '';

  // Add strength slider listener if exists
  setTimeout(() => {
    const strengthSlider = document.getElementById('param-strength');
    const strengthVal = document.getElementById('strength-val');
    if (strengthSlider && strengthVal) {
      strengthSlider.oninput = () => strengthVal.textContent = strengthSlider.value;
    }
  }, 0);

  switch(action) {
    case 'resize':
      html = '<input type="number" id="param-width" placeholder="Width" value="800"/>' +
             '<input type="number" id="param-height" placeholder="Height" value="600"/>';
      break;
    case 'crop':
      html = '<input type="number" id="param-x" placeholder="X" value="0"/>' +
             '<input type="number" id="param-y" placeholder="Y" value="0"/>' +
             '<input type="number" id="param-width" placeholder="Width" value="100"/>' +
             '<input type="number" id="param-height" placeholder="Height" value="100"/>';
      break;
    case 'rotate':
      html = '<input type="number" id="param-degrees" placeholder="Degrees" value="90"/>';
      break;
    case 'blur':
    case 'sharpen':
      html = '<input type="number" id="param-factor" placeholder="Factor (1-5)" value="2"/>';
      break;
    case 'brightness':
    case 'contrast':
      html = '<input type="number" id="param-factor" placeholder="Factor (0.1-3)" value="1.2" step="0.1"/>';
      break;
    case 'text':
      html = '<input type="text" id="param-text" placeholder="Text"/>' +
             '<input type="number" id="param-x" placeholder="X" value="10"/>' +
             '<input type="number" id="param-y" placeholder="Y" value="10"/>' +
             '<input type="number" id="param-font-size" placeholder="Font Size" value="24"/>' +
             '<input type="text" id="param-color" placeholder="Color (R,G,B)" value="255,255,255"/>';
      break;
    case 'border':
      html = '<input type="number" id="param-border-width" placeholder="Border Width" value="5"/>' +
             '<input type="text" id="param-border-color" placeholder="Color (R,G,B)" value="0,0,0"/>';
      break;
    case 'ai_edit':
      html = '<textarea id="param-prompt" placeholder="Describe what you want to change in the image (e.g., &#34;change sky to orange sunset while keeping the person&#34;)..." style="width:100%;padding:8px;margin:4px 0;border-radius:6px;background:#333;color:#fff;border:1px solid #555;min-height:80px;"></textarea>' +
             '<div style="display:flex;gap:8px;align-items:center;margin-top:8px;">' +
               '<label style="flex:1;">Strength: <span id="strength-val">0.5</span></label>' +
               '<input type="range" id="param-strength" min="0.1" max="0.7" value="0.5" step="0.05" style="flex:2;"/>' +
             '</div>' +
             '<small style="color:#888;margin-top:4px;display:block;">Lower = stays consistent with original, Higher = more dramatic changes</small>' +
             '<input type="number" id="param-seed" placeholder="Seed (optional)" style="margin-top:8px;"/>' +
             '<label style="display:flex;align-items:center;margin-top:8px;gap:8px;"><input type="checkbox" id="param-high-quality" checked/> High Quality Mode</label>';
      break;
  }

  paramsDiv.innerHTML = html;
}

document.getElementById('img-btn').onclick = openImageEditor;

document.getElementById('img-file-input').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  try {
    const data = await API.uploadImage(file);
    if (data.success) {
      currentImagePath = data.filepath;
      const preview = document.getElementById('preview-img');
      preview.src = URL.createObjectURL(file);
      document.getElementById('img-preview').style.display = 'block';
      document.getElementById('img-info').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
      document.getElementById('edit-actions').style.display = 'block';
      document.getElementById('img-upload-area').style.display = 'none';
      updateEditParams();
    } else {
      showToast(data.message);
    }
  } catch(e) {
    showToast('Upload failed: ' + e.message);
  }
};

document.getElementById('edit-action-select').onchange = updateEditParams;

document.getElementById('apply-edit-btn').onclick = async () => {
  if (!currentImagePath) {
    showToast('Please upload an image first');
    return;
  }

  const action = document.getElementById('edit-action-select').value;
  const req = { action, filepath: currentImagePath };

  const getVal = (id) => document.getElementById(id)?.value;

  switch(action) {
    case 'resize':
      req.width = parseInt(getVal('param-width')) || 800;
      req.height = parseInt(getVal('param-height')) || 600;
      break;
    case 'crop':
      req.x = parseInt(getVal('param-x')) || 0;
      req.y = parseInt(getVal('param-y')) || 0;
      req.width = parseInt(getVal('param-width')) || 100;
      req.height = parseInt(getVal('param-height')) || 100;
      break;
    case 'rotate':
      req.degrees = parseFloat(getVal('param-degrees')) || 90;
      break;
    case 'blur':
      req.radius = parseInt(getVal('param-factor')) || 2;
      break;
    case 'sharpen':
      req.factor = parseFloat(getVal('param-factor')) || 1.5;
      break;
    case 'brightness':
    case 'contrast':
      req.factor = parseFloat(getVal('param-factor')) || 1.2;
      break;
    case 'text':
      req.text = getVal('param-text') || '';
      req.x = parseInt(getVal('param-x')) || 10;
      req.y = parseInt(getVal('param-y')) || 10;
      req.font_size = parseInt(getVal('param-font-size')) || 24;
      req.color = getVal('param-color') || '255,255,255';
      break;
    case 'border':
      req.border_width = parseInt(getVal('param-border-width')) || 5;
      req.border_color = getVal('param-border-color') || '0,0,0';
      break;
    case 'ai_edit':
      const prompt = getVal('param-prompt');
      if (!prompt) {
        showToast('Please enter a prompt for AI editing');
        return;
      }
      showToast('AI editing image... (this may take a while)');
      try {
        const data = await API.aiEditImage({
          filepath: currentImagePath,
          prompt: prompt,
          strength: parseFloat(getVal('param-strength')) || 0.6,
          seed: getVal('param-seed') ? parseInt(getVal('param-seed')) : null,
          high_quality: document.getElementById('param-high-quality').checked
        });
        if (data.success) {
          const resultDiv = document.getElementById('edit-result');
          const resultImg = document.getElementById('result-img');
          resultImg.src = `/images/${data.filename}`;
          resultDiv.style.display = 'block';
          showToast('AI edit applied successfully!');
          currentImagePath = data.filename;
        } else {
          showToast(data.message);
        }
      } catch(e) {
        showToast('AI edit failed: ' + e.message);
      }
      return;
  }

  try {
    const data = await API.editImage(req);
    if (data.success) {
      const resultDiv = document.getElementById('edit-result');
      const resultImg = document.getElementById('result-img');
      resultImg.src = data.filepath;
      resultDiv.style.display = 'block';
      showToast('Edit applied successfully!');
      currentImagePath = data.filepath;
    } else {
      showToast(data.message);
    }
  } catch(e) {
    showToast('Edit failed: ' + e.message);
  }
};

document.getElementById('reset-img-btn').onclick = () => {
  document.getElementById('img-file-input').value = '';
  currentImagePath = null;
  document.getElementById('img-preview').style.display = 'none';
  document.getElementById('edit-actions').style.display = 'none';
  document.getElementById('edit-result').style.display = 'none';
  document.getElementById('img-upload-area').style.display = 'block';
};

