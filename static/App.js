// ── API ─────────────────────────────────────────────────────────────
const API = {
  async models() { return fetch('/models').then(r => r.json()); },
  async switchModel(m) { return fetch(`/models/switch?model=${encodeURIComponent(m)}`, {method:'POST'}).then(r=>r.json()); },
  async upload(file) { 
    const fd = new FormData(); 
    fd.append('file', file); 
    return fetch('/upload', {method:'POST', body:fd}).then(r=>r.json()); 
  },
  async stats() { return fetch('/memory/stats').then(r => r.json()); },
  async wipe() { return fetch('/memory/wipe', {method:'DELETE'}).then(r => r.json()); },
  async getSessions() { return fetch('/sessions').then(r => r.json()); },
  async loadSession(id) { return fetch(`/sessions/${id}`, {method:'POST'}).then(r => r.json()); },
  async newSession() { return fetch('/sessions/new', {method:'POST'}).then(r => r.json()); },
  async clearContext() { return fetch('/memory/context', {method:'DELETE'}).then(r => r.json()); },
  async getMemories() { return fetch('/memory/recent?limit=10').then(r => r.json()); },
  async draw(prompt, seed) { 
    return fetch('/draw', {
      method: 'POST', 
      headers: {'Content-Type': 'application/json'}, 
      body: JSON.stringify({prompt, seed: seed ? parseInt(seed) : null})
    }).then(r => r.json()); 
  }
};

// ── State ────────────────────────────────────────────────────────────
let currentSessionId = null;
let isStreaming = false;
let chatHistory = [];
let currentEventSource = null;

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
(async function init() {
  await loadModels();
  await refreshSessions();
  await refreshStats();
  autoResizeTextarea();
})();

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

async function loadModels() {
  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 5000);
    
    const resp = await fetch('/models', { signal: ctrl.signal });
    clearTimeout(timeout);
    
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    
    const data = await resp.json();
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
    console.error('Failed to load sessions:', e);
  }
}

async function switchSession(id) {
  try {
    currentSessionId = id;
    const data = await API.loadSession(id);
    chatWindow.innerHTML = '';
    
    if (data.messages && data.messages.length > 0) {
      welcome.classList.add('hidden');
      data.messages.forEach(msg => {
        addMessage(msg.role, msg.content);
      });
    } else {
      welcome.classList.remove('hidden');
    }
    
    await refreshSessions();
    showToast('Chat loaded');
  } catch(e) {
    showToast('Failed to load chat');
  }
}

async function createNewChat() {
  try {
    const data = await API.newSession();
    currentSessionId = data.id;
    chatWindow.innerHTML = '';
    welcome.classList.remove('hidden');
    await refreshSessions();
  } catch(e) {
    showToast('Failed to create new chat');
  }
}

function addMessage(role, text = '') {
  welcome.classList.add('hidden');
  
  const bubble = document.createElement('div');
  bubble.className = `msg ${role}`;
  
  if (role === 'bot' || role === 'assistant') {
    // Simple markdown rendering for bot messages
    bubble.innerHTML = markdownToHtml(text);
  } else {
    bubble.textContent = text;
  }
  
  chatWindow.appendChild(bubble);
  
  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  chatWindow.appendChild(time);
  
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return bubble;
}

// Simple markdown to HTML converter for code blocks
function markdownToHtml(text) {
  // Escape HTML first
  let escaped = text.replace(/[&<>"']/g, function(m) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m];
  });
  
  // Handle code blocks
  escaped = escaped.replace(/```([^`]*)```/g, function(match, code) {
    // Escape HTML in code
    const escapedCode = code.replace(/[&<>"']/g, function(m) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m];
    });
    return `<div class="code-block"><pre>${escapedCode}</pre></div>`;
  });
  
  // Handle inline code
  escaped = escaped.replace(/`([^`]*)`/g, function(match, code) {
    const escapedCode = code.replace(/[&<>"']/g, function(m) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m];
    });
    return `<code class="inline-code">${escapedCode}</code>`;
  });
  
  // Handle line breaks
  escaped = escaped.replace(/\n/g, '<br>');
  
  return escaped;
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || isStreaming) return;
  
  isStreaming = true;
  sendBtn.disabled = true;
  document.getElementById('stop-btn').style.display = 'inline-block'; // Show stop button
  
  const userBubble = addMessage('user', text);
  msgInput.value = '';
  msgInput.style.height = 'auto';
  
  const botBubble = addMessage('bot', '');
  botBubble.classList.add('streaming');
  
  try {
    const source = new EventSource(`/chat?message=${encodeURIComponent(text)}&session_id=${currentSessionId}`);
    currentEventSource = source; // Store reference to close it later
    
  source.onmessage = (e) => {
     if (e.data === '[DONE]') {
       source.close();
       isStreaming = false;
       sendBtn.disabled = false;
       document.getElementById('stop-btn').style.display = 'none'; // Hide stop button
       currentEventSource = null;
       botBubble.classList.remove('streaming');
       // Convert accumulated content to HTML after streaming completes
       botBubble.innerHTML = markdownToHtml(botBubble.textContent);
       refreshStats();
       return;
     }
     
     try {
       const { token } = JSON.parse(e.data);
       botBubble.textContent += token;
       chatWindow.scrollTop = chatWindow.scrollHeight;
     } catch(_) {}
   };
   
   source.onerror = () => {
     source.close();
     isStreaming = false;
     sendBtn.disabled = false;
     document.getElementById('stop-btn').style.display = 'none'; // Hide stop button
     currentEventSource = null;
     botBubble.classList.remove('streaming');
     if (!botBubble.textContent) {
       botBubble.textContent = 'Error: Failed to get response';
     }
   };
 } catch(e) {
   isStreaming = false;
   sendBtn.disabled = false;
   document.getElementById('stop-btn').style.display = 'none'; // Hide stop button
   currentEventSource = null;
   botBubble.classList.remove('streaming');
   botBubble.textContent = 'Error: ' + e.message;
 }
}

async function refreshStats() {
  try {
    const s = await API.stats();
    // Update stats if needed
  } catch(e) {}
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

document.getElementById('draw-btn').onclick = async () => {
  const prompt = msgInput.value.trim();
  if (!prompt) {
    showToast('Enter a prompt for image generation');
    return;
  }
  
  addMessage('user', `🎨 ${prompt}`);
  msgInput.value = '';
  
  const botBubble = addMessage('bot', 'Generating image...');
  botBubble.classList.add('streaming');
  
  try {
    const data = await API.draw(prompt);
    botBubble.classList.remove('streaming');
    botBubble.innerHTML = data.message;
    if (data.filename) {
      const img = document.createElement('img');
      img.src = `/images/${data.filename}`;
      img.style.maxWidth = '100%';
      img.style.borderRadius = '8px';
      img.style.marginTop = '12px';
      botBubble.appendChild(img);
    }
  } catch(e) {
    botBubble.classList.remove('streaming');
    botBubble.textContent = 'Error generating image';
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
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Only PDF files supported');
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
    showToast('Response stopped');
  }
};

document.getElementById('model-btn').onclick = async () => {
  try {
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 5000);
    
    const resp = await fetch('/models', { signal: ctrl.signal });
    clearTimeout(timeout);
    
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    
    const data = await resp.json();
    const models = data.models || [];
    const current = data.active;
    
    if (!models || models.length === 0) {
      showToast('No models found. Install models with: ollama pull <name>');
      return;
    }
    
    const choice = prompt(`Available models:\n${models.join('\n')}\n\nCurrent: ${current}\n\nEnter model to switch:`);
    if (choice && models.includes(choice)) {
      await API.switchModel(choice);
      activeModelLabel.textContent = choice;
      showToast(`Switched to ${choice}`);
    } else if (choice) {
      showToast(`Model not found: ${choice}`);
    }
  } catch(e) {
    console.error(e);
    showToast('Error: ' + (e.name === 'AbortError' ? 'timeout (Ollama not running?)' : e.message));
  }
};