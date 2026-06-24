// ============================================================
// STATE
// ============================================================
let videoLibrary = {};
let titleLibrary = {};
let sizeLibrary = {};
let folderLibrary = {};
let activeVideoId = "";
let activeFolderId = "";
let chatContextMode = "video";
let conversationHistory = [];
let pendingDeleteId = null;
let pendingMoveId = null;
let currentAbortController = null;
let activeIngestVideoId = null;

// ============================================================
// DOM REFS
// ============================================================
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const searchBtn = document.getElementById('search-btn');
const chatBtn = document.getElementById('chat-btn');
const cancelBtn = document.getElementById('cancel-btn');
const videoPlayer = document.getElementById('main-video');
const nowPlayingTitle = document.getElementById('now-playing-title');
const videoList = document.getElementById('video-list');
const libraryEmpty = document.getElementById('library-empty');
const resultCard = document.getElementById('result-card');
const resultContainer = document.getElementById('result-cards-container');
const ingestStatus = document.getElementById('ingest-status');
const ingestFilename = document.getElementById('ingest-filename');
const cancelIngestBtn = document.getElementById('cancel-ingest-btn');
const topkSlider = document.getElementById('topk-slider');
const topkValue = document.getElementById('topk-value');

// Settings Refs
const settingsBtn = document.getElementById('settings-btn');
const settingsOverlay = document.getElementById('settings-overlay');
const settingsProvider = document.getElementById('settings-provider');
const settingsBaseurlContainer = document.getElementById('settings-baseurl-container');
const settingsBaseurl = document.getElementById('settings-baseurl');
const settingsModelContainer = document.getElementById('settings-model-container');
const settingsModel = document.getElementById('settings-model');
const settingsApikeyContainer = document.getElementById('settings-apikey-container');
const settingsApikey = document.getElementById('settings-apikey');
const settingsCancel = document.getElementById('settings-cancel');
const settingsSave = document.getElementById('settings-save');
const uploadInput = document.getElementById('video-upload');
const modalOverlay = document.getElementById('modal-overlay');
const modalTitle = document.getElementById('modal-title');
const modalMessage = document.getElementById('modal-message');
const modalFolderSelect = document.getElementById('modal-folder-select');
const modalInput = document.getElementById('modal-input');
const modalConfirm = document.getElementById('modal-confirm');
const modalCancel = document.getElementById('modal-cancel');

// ============================================================
// HELPERS
// ============================================================
function setInputState(disabled) {
    userInput.disabled = disabled;
    searchBtn.disabled = disabled;
    chatBtn.disabled = disabled;
    if (disabled) {
        cancelBtn.style.display = 'inline-block';
        searchBtn.style.display = 'none';
        chatBtn.style.display = 'none';
        userInput.placeholder = "Processing... Please wait.";
        userInput.style.opacity = "0.5";
    } else {
        cancelBtn.style.display = 'none';
        searchBtn.style.display = 'inline-block';
        chatBtn.style.display = 'inline-block';
        userInput.placeholder = "Ask a question or search for something...";
        userInput.style.opacity = "1";
        setTimeout(() => userInput.focus(), 100);
    }
}

cancelBtn.addEventListener('click', () => {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
});

cancelIngestBtn.addEventListener('click', async () => {
    if (activeIngestVideoId) {
        cancelIngestBtn.disabled = true;
        cancelIngestBtn.textContent = 'Cancelling...';
        try {
            await fetch('/api/ingest/' + encodeURIComponent(activeIngestVideoId) + '/cancel', { method: 'POST' });
        } catch (e) {
            console.error('Failed to send cancel request', e);
        }
    }
});

function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function addMessage(sender, text, isThinking = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender === 'user' ? 'msg-user' : 'msg-ai');
    if (isThinking) msgDiv.classList.add('thinking');
    msgDiv.textContent = text;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv;
}

function removeElement(el) { el && el.remove(); }

// ============================================================
// LIBRARY — Boot & Render
// ============================================================
async function bootSystem() {
    try {
        const res = await fetch('/library');
        const data = await res.json();

        videoLibrary = data.videoLibrary || {};
        titleLibrary = data.titleLibrary || {};
        sizeLibrary = data.sizeLibrary || {};
        folderLibrary = data.folderLibrary || {};

        renderLibrary();

        const ids = Object.keys(videoLibrary);
        if (ids.length > 0 && !activeVideoId) {
            setActiveVideo(ids[0]);
        }

        // Check for background ingestions
        const activeRes = await fetch('/api/ingest/active');
        const activeData = await activeRes.json();
        if (activeData.active_jobs && activeData.active_jobs.length > 0) {
            const videoId = activeData.active_jobs[0];
            showProgress();
            ingestStatus.textContent = `🔄 Reconnecting to active ingestion: ${videoId}`;
            isIngesting = true;
            startIngestStreamForQueue(videoId, "", videoId);
        }
    } catch (e) {
        nowPlayingTitle.textContent = 'Error: Backend disconnected.';
        console.error('Boot failed:', e);
    }
}

function renderLibrary() {
    Array.from(videoList.children).forEach(c => {
        if (c.id !== 'library-empty') c.remove();
    });

    const ids = Object.keys(videoLibrary);
    libraryEmpty.style.display = ids.length === 0 ? 'flex' : 'none';

    const folders = {};
    ids.forEach(id => {
        let folder = folderLibrary[id] || "Uncategorized";
        if (folder === "." || folder === "") folder = "Uncategorized";
        if (!folders[folder]) folders[folder] = [];
        folders[folder].push(id);
    });

    Object.keys(folders).sort().forEach(folderName => {
        const folderDiv = document.createElement('div');
        folderDiv.className = 'folder-group';

        const folderHeader = document.createElement('div');
        folderHeader.className = 'folder-header';
        folderHeader.innerHTML = `📁 ${folderName} <span class="folder-count">(${folders[folderName].length})</span>`;
        folderDiv.appendChild(folderHeader);

        const folderContent = document.createElement('div');
        folderContent.className = 'folder-content';

        folders[folderName].forEach(id => {
            const card = document.createElement('div');
            card.className = 'video-card' + (id === activeVideoId ? ' active' : '');
            card.id = `vcard-${id}`;

            card.innerHTML = `
                <img class="card-thumb" src="/thumbnail/${id}" alt="" loading="lazy"
                     onerror="this.outerHTML='<div class=\\'card-thumb-placeholder\\'>▶</div>'">
                <div class="card-info">
                    <div class="card-title" title="${titleLibrary[id]}">${titleLibrary[id]}</div>
                    <div class="card-size">${sizeLibrary[id] ? formatBytes(sizeLibrary[id]) : ''}</div>
                </div>
                <button class="card-move" title="Move to folder" data-id="${id}">📁</button>
                <button class="card-reingest" title="Re-ingest video" data-id="${id}">🔄</button>
                <button class="card-delete" title="Delete video" data-id="${id}">✕</button>
            `;

            card.draggable = true;
            card.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', id);
                e.dataTransfer.effectAllowed = 'move';
            });

            card.addEventListener('click', (e) => {
                if (e.target.classList.contains('card-delete')) return;
                if (e.target.classList.contains('card-reingest')) return;
                if (e.target.classList.contains('card-move')) return;
                setActiveVideo(id);
            });

            card.querySelector('.card-reingest').addEventListener('click', (e) => {
                e.stopPropagation();
                handleReingest(id);
            });

            card.querySelector('.card-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                openDeleteModal(id);
            });

            card.querySelector('.card-move').addEventListener('click', (e) => {
                e.stopPropagation();
                openMoveModal(id);
            });

            folderContent.appendChild(card);
        });

        folderHeader.addEventListener('click', () => {
            folderContent.style.display = folderContent.style.display === 'none' ? 'block' : 'none';
        });

        folderHeader.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        });

        folderHeader.addEventListener('dragenter', (e) => {
            e.preventDefault();
            folderHeader.classList.add('drag-over');
        });

        folderHeader.addEventListener('dragleave', (e) => {
            folderHeader.classList.remove('drag-over');
        });

        folderHeader.addEventListener('drop', async (e) => {
            e.preventDefault();
            folderHeader.classList.remove('drag-over');
            const videoId = e.dataTransfer.getData('text/plain');
            if (videoId) {
                await moveVideoToFolder(videoId, folderName === "Uncategorized" ? "" : folderName);
            }
        });

        folderDiv.appendChild(folderContent);
        videoList.insertBefore(folderDiv, libraryEmpty);
    });
}

function updateChatContextUI() {
    const chatControls = document.getElementById('chat-controls');
    let toggleDiv = document.getElementById('chat-context-toggle');

    if (!toggleDiv) {
        toggleDiv = document.createElement('div');
        toggleDiv.id = 'chat-context-toggle';
        toggleDiv.innerHTML = `
            <label><input type="radio" name="chat_ctx" value="video" checked> Current File</label>
            <label style="margin-left:8px"><input type="radio" name="chat_ctx" value="folder"> Entire Folder</label>
        `;
        toggleDiv.style.marginBottom = "10px";
        toggleDiv.style.fontSize = "12px";

        toggleDiv.addEventListener('change', (e) => {
            chatContextMode = e.target.value;
            conversationHistory = [];
            if (chatContextMode === 'folder') {
                chatHistory.innerHTML = `<div class="message msg-ai">👋 Changed context to "Folder: ${activeFolderId}". Ask me anything about this folder!</div>`;
            } else {
                let mediaTypeStr = "video";
                if (activeVideoId && videoLibrary[activeVideoId]) {
                    const ext = videoLibrary[activeVideoId].split('.').pop().toLowerCase();
                    if (ext === "pdf") mediaTypeStr = "document";
                    else if (["jpg", "jpeg", "png", "webp"].includes(ext)) mediaTypeStr = "image";
                    else if (["mp3", "wav", "m4a"].includes(ext)) mediaTypeStr = "audio file";
                }
                chatHistory.innerHTML = `<div class="message msg-ai">👋 Changed context to "${titleLibrary[activeVideoId] || activeVideoId}". Ask me anything about this ${mediaTypeStr}!</div>`;
            }
        });

        chatControls.insertBefore(toggleDiv, document.getElementById('input-row'));
    }

    const folderRadioLabel = toggleDiv.querySelector('input[value="folder"]').parentElement;
    if (activeFolderId && activeFolderId !== "Uncategorized") {
        folderRadioLabel.style.display = "inline";
    } else {
        folderRadioLabel.style.display = "none";
        toggleDiv.querySelector('input[value="video"]').checked = true;
    }
}

function setActiveVideo(id) {
    activeVideoId = id;
    let folderRaw = folderLibrary[id] || "Uncategorized";
    activeFolderId = (folderRaw === "." || folderRaw === "") ? "Uncategorized" : folderRaw;
    conversationHistory = [];

    // Reset chat context
    chatContextMode = "video";
    document.querySelectorAll('.tab').forEach(x => {
        x.classList.remove('active');
        if (x.dataset.tab === 'video') x.classList.add('active');
    });

    // Reset Chat History visually when switching context
    let mediaTypeStr = "video";
    if (id && videoLibrary[id]) {
        const ext = videoLibrary[id].split('.').pop().toLowerCase();
        if (ext === "pdf") mediaTypeStr = "document";
        else if (["jpg", "jpeg", "png", "webp"].includes(ext)) mediaTypeStr = "image";
        else if (["mp3", "wav", "m4a"].includes(ext)) mediaTypeStr = "audio file";
    }
    chatHistory.innerHTML = `<div class="message msg-ai">👋 Changed context to "${titleLibrary[id] || id}". Ask me anything about this ${mediaTypeStr}!</div>`;

    // Update sidebar highlight
    document.querySelectorAll('.video-card').forEach(c => c.classList.remove('active'));
    const card = document.getElementById(`vcard-${id}`);
    if (card) card.classList.add('active');

    // Update player based on media type
    nowPlayingTitle.textContent = titleLibrary[id] || id;
    const mediaUrl = videoLibrary[id];
    const ext = mediaUrl.split('.').pop().toLowerCase();

    const mainImage = document.getElementById('main-image');
    const mainPdf = document.getElementById('main-pdf');

    if (ext === "pdf") {
        // PDF Mode
        videoPlayer.style.display = 'none';
        videoPlayer.pause();
        mainImage.style.display = 'none';
        mainPdf.style.display = 'block';
        mainPdf.src = mediaUrl;
    } else if (["jpg", "jpeg", "png", "webp"].includes(ext)) {
        // Image Mode
        videoPlayer.style.display = 'none';
        videoPlayer.pause();
        mainPdf.style.display = 'none';
        mainImage.style.display = 'block';
        mainImage.src = mediaUrl;
    } else {
        // Video / Audio Mode
        mainImage.style.display = 'none';
        mainPdf.style.display = 'none';
        videoPlayer.style.display = 'block';
        videoPlayer.src = mediaUrl;
        videoPlayer.load();
        videoPlayer.play().catch(e => console.log("Autoplay blocked by browser:", e));
    }

    updateChatContextUI();
} // Note: result cards are NOT cleared here — user must click ✕ to dismiss them


// ============================================================
// INTENT A — SEARCH & SEEK
// ============================================================
async function handleSearch() {
    const query = userInput.value.trim();
    if (!query) return;

    addMessage('user', query);
    userInput.value = '';
    setInputState(true);

    const thinking = addMessage('ai', '🧠 Searching tri-modal vector database', true);
    const topK = parseInt(topkSlider.value, 10);

    currentAbortController = new AbortController();

    try {
        const res = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK }),
            signal: currentAbortController.signal
        });
        const data = await res.json();
        removeElement(thinking);

        if (data.results && data.results.length > 0) {
            renderResultCards(data.results);
            const best = data.results[0];

            // Guard: only switch video if it exists in the current library
            if (!videoLibrary[best.video_id]) {
                addMessage('ai', `⚠️ Found results but video "${best.video_id}" is not in your library. Try re-ingesting it.`);
            } else {
                // Switch video if needed
                if (activeVideoId !== best.video_id) {
                    setActiveVideo(best.video_id);
                }

                addMessage('ai', `🎯 Found ${data.results.length} result${data.results.length > 1 ? 's' : ''}. Jumping to ${formatTime(best.start_timestamp)} in "${titleLibrary[best.video_id] || best.video_id}".`);

                const ext = videoLibrary[best.video_id].split('.').pop().toLowerCase();
                if (ext === "pdf") {
                    const mainPdf = document.getElementById('main-pdf');
                    mainPdf.src = videoLibrary[best.video_id] + "#page=" + Math.floor(best.start_timestamp) + "&search=" + encodeURIComponent(best.llm_optimized_query);
                } else if (!["jpg", "jpeg", "png", "webp"].includes(ext)) {
                    videoPlayer.onloadedmetadata = () => {
                        videoPlayer.currentTime = Math.max(0, best.start_timestamp - 1);
                        videoPlayer.play().catch(e => console.log("Play error:", e));
                    };
                    videoPlayer.currentTime = Math.max(0, best.start_timestamp - 1);
                    videoPlayer.play().catch(e => console.log("Play error:", e));
                }
            }
        } else {
            resultCard.classList.remove('visible');
            addMessage('ai', data.message || '❌ No relevant segments found for that query.');
        }
    } catch (e) {
        removeElement(thinking);
        if (e.name === 'AbortError') {
            addMessage('ai', '🛑 Search canceled.');
        } else {
            addMessage('ai', '⚠️ Error connecting to the backend /search API.');
        }
    } finally {
        currentAbortController = null;
        setInputState(false);
    }
}

// ============================================================
// RESULT CARD — CLOSE BUTTON & LABEL
// ============================================================
const resultCardClose = document.getElementById('result-card-close');
const resultCardLabel = document.getElementById('result-card-label');

resultCardClose.addEventListener('click', () => {
    resultCard.classList.remove('visible');
    resultContainer.innerHTML = '';
});

function renderResultCards(results) {
    resultContainer.innerHTML = '';
    resultCardLabel.textContent = `Search Results — ${results.length}`;
    results.forEach((r, i) => {
        const item = document.createElement('div');
        item.className = 'result-item' + (i === 0 ? ' selected' : '');

        item.innerHTML = `
            <div class="result-meta">
                <span class="result-badge rb-video">🎬 ${titleLibrary[r.video_id] || r.video_id}</span>
                <span class="result-badge rb-time">⏱ ${formatTime(r.start_timestamp)} → ${formatTime(r.end_timestamp)}</span>
                <span class="result-badge rb-score">📊 ${(r.hybrid_rrf_score * 100).toFixed(1)}%</span>
                <span class="result-badge rb-query">🔍 "${r.llm_optimized_query}"</span>
            </div>
            <div class="result-transcript">${r.matched_transcript || 'No transcript available.'}</div>
        `;

        item.addEventListener('click', () => {
            document.querySelectorAll('.result-item').forEach(el => el.classList.remove('selected'));
            item.classList.add('selected');

            if (activeVideoId !== r.video_id) setActiveVideo(r.video_id);

            const ext = videoLibrary[r.video_id].split('.').pop().toLowerCase();
            if (ext === "pdf") {
                const mainPdf = document.getElementById('main-pdf');
                mainPdf.src = videoLibrary[r.video_id] + "#page=" + Math.floor(r.start_timestamp) + "&search=" + encodeURIComponent(r.llm_optimized_query);
            } else if (!["jpg", "jpeg", "png", "webp"].includes(ext)) {
                videoPlayer.onloadedmetadata = () => {
                    videoPlayer.currentTime = Math.max(0, r.start_timestamp - 1);
                    videoPlayer.play().catch(e => console.log("Play error:", e));
                };
                videoPlayer.currentTime = Math.max(0, r.start_timestamp - 1);
                videoPlayer.play().catch(e => console.log("Play error:", e));
            }
        });

        resultContainer.appendChild(item);
    });

    resultCard.classList.add('visible');
}

// ============================================================
// INTENT B — CHAT & SUMMARIZE
// ============================================================
async function handleChat() {
    const query = userInput.value.trim();
    if (!query) return;
    if (!activeVideoId) {
        addMessage('ai', '⚠️ Please select a video from the library first.');
        return;
    }

    let targetIds = [activeVideoId];
    let contextName = titleLibrary[activeVideoId];

    if (chatContextMode === "folder" && activeFolderId && activeFolderId !== "Uncategorized") {
        targetIds = Object.keys(folderLibrary).filter(id => folderLibrary[id] === activeFolderId);
        contextName = `Folder: ${activeFolderId} (${targetIds.length} videos)`;
    }

    addMessage('user', query);
    userInput.value = '';

    const previousHistory = [...conversationHistory];
    conversationHistory.push({ role: 'user', content: query });
    if (conversationHistory.length > 6) conversationHistory = conversationHistory.slice(-6);

    setInputState(true);
    const thinking = addMessage('ai', `🤖 Analyzing "${contextName}"`, true);

    currentAbortController = new AbortController();

    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                target_video_ids: targetIds,
                chat_history: previousHistory
            }),
            signal: currentAbortController.signal
        });

        if (!res.ok) throw new Error('Network error');

        let aiMsgElement = null;

        // Stream Reader
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;
        let fullAnswer = '';
        let answerStarted = false;

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                fullAnswer += chunk;

                if (!answerStarted && fullAnswer.includes('---')) {
                    // The thought stream is over, spawn the real answer bubble
                    answerStarted = true;
                    removeElement(thinking);
                    aiMsgElement = document.createElement('div');
                    aiMsgElement.classList.add('message', 'msg-ai');
                    chatHistory.appendChild(aiMsgElement);
                }

                if (!answerStarted) {
                    // Route the backend logs into the temporary grey bubble
                    thinking.textContent = fullAnswer.trimStart();
                } else {
                    // Route the real answer into the white bubble
                    aiMsgElement.textContent = fullAnswer.split('---').pop().trimStart();
                }

                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }

        let finalHistoryText = fullAnswer;
        if (fullAnswer.includes('---')) {
            finalHistoryText = fullAnswer.split('---').pop().trim();
        }
        conversationHistory.push({ role: 'ai', content: finalHistoryText });
    } catch (e) {
        removeElement(thinking);
        if (e.name === 'AbortError') {
            addMessage('ai', '🛑 Chat generation canceled.');
        } else {
            addMessage('ai', '⚠️ Error connecting to the backend /chat API.');
        }
    } finally {
        currentAbortController = null;
        setInputState(false);
    }
}

// ============================================================
// PROGRESS BAR HELPERS
// ============================================================
const progressBar = document.getElementById('ingest-progress-bar');
const phaseSteps = [1, 2, 3, 4].map(n => document.getElementById(`phase-${n}`));

function setPhase(n) {
    // n = 0 means reset, 1-4 means that phase is active, -1 means all done
    phaseSteps.forEach((step, i) => {
        step.classList.remove('active', 'done');
        if (n === -1) { step.classList.add('done'); }
        else if (i + 1 < n) { step.classList.add('done'); }
        else if (i + 1 === n) { step.classList.add('active'); }
    });
}

function showProgress() {
    progressBar.classList.add('visible');
    setPhase(1);
}

function hideProgress() {
    progressBar.classList.remove('visible');
    setPhase(0);
}

// ============================================================
// UPLOAD & INGEST (WITH QUEUE SUPPORT)
// ============================================================
const folderUploadInput = document.getElementById('folder-upload');
let ingestQueue = [];
let isIngesting = false;

function handleFilesSelected(files) {
    if (!files || files.length === 0) return;

    const validExts = ['.mp4', '.mkv', '.mp3', '.wav', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.pdf'];
    const validFiles = Array.from(files).filter(f => {
        const name = f.name.toLowerCase();
        return validExts.some(ext => name.endsWith(ext));
    });

    if (validFiles.length === 0) {
        ingestStatus.textContent = '❌ No valid media files found (video/audio/image/pdf).';
        return;
    }

    ingestQueue.push(...validFiles);

    uploadInput.value = '';
    if (folderUploadInput) folderUploadInput.value = '';

    if (!isIngesting) {
        processIngestQueue();
    } else {
        ingestStatus.textContent = `⏳ Added ${validFiles.length} file(s) to queue. (${ingestQueue.length} total pending)`;
    }
}

uploadInput.addEventListener('change', (e) => handleFilesSelected(e.target.files));
if (folderUploadInput) folderUploadInput.addEventListener('change', (e) => handleFilesSelected(e.target.files));

async function processIngestQueue() {
    if (ingestQueue.length === 0) {
        isIngesting = false;
        setPhase(-1);
        ingestStatus.textContent = '✅ All files ingested! Refreshing library…';
        setTimeout(async () => {
            await bootSystem();
            ingestStatus.textContent = '';
            hideProgress();
        }, 1500);
        return;
    }

    isIngesting = true;
    const file = ingestQueue.shift();

    ingestStatus.textContent = `📤 Uploading "${file.name}"… (${ingestQueue.length} remaining in queue)`;
    showProgress();

    const formData = new FormData();
    formData.append('file', file);

    if (file.webkitRelativePath) {
        const parts = file.webkitRelativePath.split('/');
        if (parts.length > 1) {
            formData.append('folder_path', parts.slice(0, -1).join('/'));
        }
    }

    try {
        const uploadRes = await fetch('/upload', { method: 'POST', body: formData });

        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            ingestStatus.textContent = `❌ ${err.detail} (${file.name})`;
            setTimeout(processIngestQueue, 2000);
            return;
        }

        const uploadData = await uploadRes.json();
        ingestStatus.textContent = `⚙️ Starting AI engine for "${file.name}"…`;

        // 1. Trigger the background ML job
        await fetch(`/api/ingest/start?video_id=${encodeURIComponent(uploadData.video_id)}&file_path=${encodeURIComponent(uploadData.file_path)}`, { method: 'POST' });

        // 2. Attach the UI stream listener
        startIngestStreamForQueue(uploadData.video_id, uploadData.file_path, file.name);

    } catch (err) {
        ingestStatus.textContent = `❌ Upload failed for "${file.name}".`;
        console.error(err);
        setTimeout(processIngestQueue, 2000);
    }
}

function startIngestStreamForQueue(videoId, filePath, fileName) {
    activeIngestVideoId = videoId;
    cancelIngestBtn.style.display = 'inline-block';
    cancelIngestBtn.disabled = false;
    // Removed textContent overwrite to keep it as small icon

    const streamUrl = `/ingest/${videoId}?file_path=${encodeURIComponent(filePath)}`;
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
        const msg = event.data;
        ingestFilename.textContent = fileName;
        ingestStatus.textContent = msg;

        if (msg.includes('[1/4]')) setPhase(1);
        else if (msg.includes('[2/4]')) setPhase(2);
        else if (msg.includes('[3/4]')) setPhase(3);
        else if (msg.includes('[4/4]')) setPhase(4);

        if (msg.includes('[COMPLETE]') || msg.includes('[CANCELED]')) {
            eventSource.close();
            setPhase(-1);
            cancelIngestBtn.style.display = 'none';
            activeIngestVideoId = null;

            if (msg.includes('[CANCELED]')) {
                ingestStatus.textContent = `🛑 Canceled.`;
                // Refresh library to remove the canceled file
                setTimeout(async () => {
                    await bootSystem();
                    ingestFilename.textContent = '';
                    ingestStatus.textContent = '';
                    hideProgress();
                    setTimeout(processIngestQueue, 500);
                }, 1500);
            } else {
                ingestStatus.textContent = `✅ Done!`;
                setTimeout(() => {
                    ingestFilename.textContent = '';
                    processIngestQueue();
                }, 1500);
            }
        }
    };

    eventSource.onerror = () => {
        ingestStatus.textContent = `⚠️ Stream disconnected for "${fileName}".`;
        eventSource.close();
        cancelIngestBtn.style.display = 'none';
        activeIngestVideoId = null;
        setTimeout(processIngestQueue, 2000);
    };
}

// Keeping original startIngestStream for handleReingest backwards compatibility
function startIngestStream(videoId, filePath) {
    activeIngestVideoId = videoId;
    cancelIngestBtn.style.display = 'inline-block';
    cancelIngestBtn.disabled = false;

    const streamUrl = `/ingest/${videoId}?file_path=${encodeURIComponent(filePath)}`;
    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
        const msg = event.data;
        ingestStatus.textContent = msg;

        if (msg.includes('[1/4]')) setPhase(1);
        else if (msg.includes('[2/4]')) setPhase(2);
        else if (msg.includes('[3/4]')) setPhase(3);
        else if (msg.includes('[4/4]')) setPhase(4);

        if (msg.includes('[COMPLETE]') || msg.includes('[CANCELED]')) {
            eventSource.close();
            setPhase(-1);
            cancelIngestBtn.style.display = 'none';
            activeIngestVideoId = null;

            if (msg.includes('[CANCELED]')) {
                ingestStatus.textContent = '🛑 Canceled. Refreshing library…';
            } else {
                ingestStatus.textContent = '✅ Done! Refreshing library…';
            }

            setTimeout(async () => {
                await bootSystem();
                ingestFilename.textContent = '';
                ingestStatus.textContent = '';
                hideProgress();
            }, 1500);
        }
    };

    eventSource.onerror = () => {
        ingestStatus.textContent = '⚠️ Connection lost.';
        eventSource.close();
        cancelIngestBtn.style.display = 'none';
        activeIngestVideoId = null;
        setTimeout(() => {
            ingestFilename.textContent = '';
            ingestStatus.textContent = '';
            hideProgress();
        }, 2000);
    };
}

// ============================================================
// RE-INGEST (delete vectors only, then re-run pipeline)
// ============================================================
async function handleReingest(videoId) {
    const title = titleLibrary[videoId] || videoId;
    if (!confirm(`Re-ingest "${title}"? This will delete its existing vectors and re-run the full AI pipeline on the file.`)) return;

    ingestStatus.textContent = `🗑 Clearing vectors for "${title}"…`;
    showProgress();

    try {
        // Delete vectors only (keep the file: delete_file=false)
        const delRes = await fetch(`/video/${encodeURIComponent(videoId)}?delete_file=false`, {
            method: 'DELETE'
        });
        const delData = await delRes.json();

        if (!delData.success) throw new Error('Vector deletion failed');

        ingestStatus.textContent = `⚙️ Re-ingesting "${title}"…`;

        // Find the file path from the library
        const filePath = videoLibrary[videoId]?.replace(/^\//, '') || `media/${videoId}.mp4`;

        await fetch(`/api/ingest/start?video_id=${encodeURIComponent(videoId)}&file_path=${encodeURIComponent(filePath)}`, { method: 'POST' });

        startIngestStream(videoId, filePath);

    } catch (err) {
        ingestStatus.textContent = '❌ Re-ingest failed.';
        hideProgress();
        console.error(err);
    }
}

// ============================================================
// MODALS (DELETE & MOVE)
// ============================================================
async function moveVideoToFolder(idToMove, newFolder) {
    try {
        const res = await fetch(`/video/${encodeURIComponent(idToMove)}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_folder: newFolder })
        });
        const data = await res.json();

        if (data.success) {
            addMessage('ai', `📁 Moved "${titleLibrary[idToMove] || idToMove}" to folder "${newFolder || 'Uncategorized'}".`);
            await bootSystem();
        } else {
            addMessage('ai', `⚠️ Move failed: ${data.error || 'Unknown error'}`);
        }
    } catch (e) {
        addMessage('ai', '⚠️ Failed to move the video. Check the backend.');
    }
}

function openDeleteModal(videoId) {
    pendingDeleteId = videoId;
    pendingMoveId = null;
    modalTitle.textContent = "🗑 Delete Video";
    modalMessage.textContent = `Delete "${titleLibrary[videoId] || videoId}"? This removes all vectors from the database and deletes the media file from disk. This cannot be undone.`;
    modalFolderSelect.style.display = "none";
    modalInput.style.display = "none";
    modalInput.value = "";
    modalConfirm.textContent = "Delete";
    modalOverlay.classList.add('open');
}

function openMoveModal(videoId) {
    pendingMoveId = videoId;
    pendingDeleteId = null;
    modalTitle.textContent = "📁 Move Video";
    modalMessage.textContent = `Move "${titleLibrary[videoId] || videoId}" to a folder:`;

    // Populate select
    modalFolderSelect.innerHTML = '<option value="">[Root / Uncategorized]</option>';
    const existingFolders = [...new Set(Object.values(folderLibrary))].filter(f => f && f !== "." && f !== "Uncategorized");
    existingFolders.sort().forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = `📁 ${f}`;
        modalFolderSelect.appendChild(opt);
    });
    const newOpt = document.createElement('option');
    newOpt.value = "__NEW__";
    newOpt.textContent = "➕ Create New Folder...";
    modalFolderSelect.appendChild(newOpt);

    let currentFolder = folderLibrary[videoId] || "";
    if (currentFolder === "Uncategorized" || currentFolder === ".") currentFolder = "";

    if (existingFolders.includes(currentFolder)) {
        modalFolderSelect.value = currentFolder;
        modalInput.style.display = "none";
    } else if (currentFolder === "") {
        modalFolderSelect.value = "";
        modalInput.style.display = "none";
    } else {
        modalFolderSelect.value = "__NEW__";
        modalInput.style.display = "block";
        modalInput.value = currentFolder;
    }

    modalFolderSelect.style.display = "block";
    modalConfirm.textContent = "Move";
    modalOverlay.classList.add('open');
}

modalFolderSelect.addEventListener('change', (e) => {
    if (e.target.value === "__NEW__") {
        modalInput.style.display = "block";
        modalInput.value = "";
        modalInput.focus();
    } else {
        modalInput.style.display = "none";
    }
});

function closeModal() {
    modalOverlay.classList.remove('open');
    pendingDeleteId = null;
    pendingMoveId = null;
    modalInput.value = "";
    modalInput.style.display = "none";
    modalFolderSelect.style.display = "none";
}

modalCancel.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) closeModal(); });

modalConfirm.addEventListener('click', async () => {
    if (pendingDeleteId) {
        // DELETE LOGIC
        const idToDelete = pendingDeleteId;
        closeModal();

        const card = document.getElementById(`vcard-${idToDelete}`);
        if (card) card.style.opacity = '0.4';

        try {
            const res = await fetch(`/video/${encodeURIComponent(idToDelete)}?delete_file=true`, {
                method: 'DELETE'
            });
            const data = await res.json();

            if (data.success) {
                delete videoLibrary[idToDelete];
                delete titleLibrary[idToDelete];
                delete sizeLibrary[idToDelete];
                delete folderLibrary[idToDelete];

                if (activeVideoId === idToDelete) {
                    activeVideoId = '';
                    videoPlayer.src = '';
                    nowPlayingTitle.textContent = 'No video selected';
                    resultCard.classList.remove('visible');
                    const chatContextToggle = document.getElementById('chat-context-toggle');
                    if (chatContextToggle) chatContextToggle.style.display = 'none';
                }

                addMessage('ai', `🗑 Deleted "${data.video_id}" — removed ${data.vectors_deleted} vectors.`);
                renderLibrary();
            }
        } catch (e) {
            if (card) card.style.opacity = '1';
            addMessage('ai', '⚠️ Failed to delete the video. Check the backend.');
        }
    } else if (pendingMoveId) {
        // MOVE LOGIC
        const idToMove = pendingMoveId;
        let newFolder = modalFolderSelect.value;
        if (newFolder === "__NEW__") {
            newFolder = modalInput.value.trim();
        }
        closeModal();
        await moveVideoToFolder(idToMove, newFolder);
    }
});

// ============================================================
// CONTROLS
// ============================================================
topkSlider.addEventListener('input', () => {
    topkValue.textContent = topkSlider.value;
});

searchBtn.addEventListener('click', handleSearch);
chatBtn.addEventListener('click', handleChat);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleChat();
    }
});

// ============================================================
// SETTINGS MODAL LOGIC
// ============================================================
function updateSettingsFields() {
    const val = settingsProvider.value;
    settingsModelContainer.style.display = 'block'; // ALWAYS show Model Name

    if (val === 'local' || val === 'ollama' || val === 'custom') {
        settingsBaseurlContainer.style.display = 'block';
        settingsApikeyContainer.style.display = (val === 'custom') ? 'block' : 'none';

        // Auto-fill defaults if empty
        if (val === 'ollama' && (!settingsBaseurl.value || settingsBaseurl.value.includes('1234') || settingsBaseurl.value.includes('api.openai.com'))) {
            settingsBaseurl.value = 'http://host.docker.internal:11434/v1';
            settingsModel.value = 'llama3';
        } else if (val === 'local' && (!settingsBaseurl.value || settingsBaseurl.value.includes('11434') || settingsBaseurl.value.includes('api.openai.com'))) {
            settingsBaseurl.value = 'http://host.docker.internal:1234/v1';
            settingsModel.value = 'Llama-3.2-3B-Instruct-Q4_K_M';
        } else if (val === 'custom' && (!settingsBaseurl.value || settingsBaseurl.value.includes('host.docker.internal') || settingsBaseurl.value.includes('127.0.0.1') || settingsBaseurl.value.includes('localhost'))) {
            settingsBaseurl.value = 'https://api.openai.com/v1';
            settingsModel.value = 'gpt-4o-mini';
        }
    } else {
        // OpenAI or Gemini
        settingsBaseurlContainer.style.display = 'none';
        settingsApikeyContainer.style.display = 'block';

        if (val === 'openai' && (!settingsModel.value || settingsModel.value.includes('llama') || settingsModel.value.includes('gemini'))) {
            settingsModel.value = 'gpt-4o-mini';
        } else if (val === 'gemini' && (!settingsModel.value || settingsModel.value.includes('llama') || settingsModel.value.includes('gpt'))) {
            settingsModel.value = 'gemini-1.5-flash-latest';
        }
    }
}

settingsProvider.addEventListener('change', updateSettingsFields);

settingsBtn.addEventListener('click', async () => {
    try {
        const res = await fetch('/api/settings/llm');
        if (res.ok) {
            const data = await res.json();
            settingsProvider.value = data.provider;
            settingsBaseurl.value = data.base_url;
            settingsModel.value = data.model;
            settingsApikey.value = data.api_key;
            updateSettingsFields();
            settingsOverlay.style.display = 'flex';
        }
    } catch (e) {
        console.error("Failed to fetch settings", e);
    }
});

settingsCancel.addEventListener('click', () => {
    settingsOverlay.style.display = 'none';
});

settingsSave.addEventListener('click', async () => {
    const payload = {
        provider: settingsProvider.value,
        base_url: settingsBaseurl.value.trim(),
        model: settingsModel.value.trim(),
        api_key: settingsApikey.value.trim()
    };

    const originalText = settingsSave.textContent;
    settingsSave.textContent = 'Saving...';

    try {
        const res = await fetch('/api/settings/llm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            settingsOverlay.style.display = 'none';
            addMessage('ai', `⚙️ Settings updated! Switched LLM provider to ${payload.provider.toUpperCase()}.`);
        } else {
            alert("Failed to update settings.");
        }
    } catch (e) {
        console.error(e);
        alert("Network error.");
    } finally {
        settingsSave.textContent = originalText;
    }
});

settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) {
        settingsOverlay.style.display = 'none';
    }
});

// ============================================================
// BOOT
// ============================================================
bootSystem();
