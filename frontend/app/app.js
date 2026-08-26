// app.js

// ===============================
// تنظیمات
// ===============================
const CONFIG = {
    BASE_URL: 'http://localhost:8000/api',
    TOKEN: localStorage.getItem('api_token') || '',
    SESSION_ID: 1,
    POLL_INTERVAL: 3000, // میلی‌ثانیه
};

// ===============================
// توابع عمومی
// ===============================
function showToast(title, message, type = 'info') {
    const toastEl = document.getElementById('toast');
    const titleEl = document.getElementById('toastTitle');
    const messageEl = document.getElementById('toastMessage');
    
    titleEl.textContent = title;
    messageEl.textContent = message;
    
    toastEl.className = `toast bg-${type} text-white`;
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function getAuthHeaders() {
    return {
        'Authorization': `Bearer ${CONFIG.TOKEN}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };
}

// ===============================
// مدیریت تَب‌ها
// ===============================
document.querySelectorAll('[data-tab]').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Active class
        document.querySelectorAll('[data-tab]').forEach(l => l.classList.remove('active'));
        this.classList.add('active');
        
        // Show tab
        const tabName = this.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.getElementById(`tab-${tabName}`).classList.add('active');
        
        // Load data if needed
        if (tabName === 'documents') loadDocuments();
        if (tabName === 'history') loadHistory();
    });
});

// ===============================
// آپلود فایل
// ===============================
let selectedFiles = [];

// Drop Zone
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    const validFiles = Array.from(files).filter(f => f.type === 'application/pdf');
    
    if (validFiles.length === 0) {
        showToast('خطا', 'فایل‌های PDF معتبر انتخاب کنید', 'danger');
        return;
    }
    
    selectedFiles = [...selectedFiles, ...validFiles];
    renderFileList();
    document.getElementById('uploadBtn').disabled = false;
}

function renderFileList() {
    const container = document.getElementById('fileList');
    container.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item">
            <div>
                <span class="file-name"><i class="fas fa-file-pdf text-danger me-1"></i> ${file.name}</span>
                <span class="file-size ms-2">${formatFileSize(file.size)}</span>
            </div>
            <button class="remove-file" onclick="removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
    if (selectedFiles.length === 0) {
        document.getElementById('uploadBtn').disabled = true;
    }
}

// Upload
document.getElementById('uploadBtn').addEventListener('click', uploadFiles);

async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    const btn = document.getElementById('uploadBtn');
    const progress = document.getElementById('uploadProgress');
    const bar = progress.querySelector('.progress-bar');
    const status = document.getElementById('uploadStatus');
    
    btn.disabled = true;
    progress.classList.remove('d-none');
    
    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('files[]', file));
    
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/sessions/${CONFIG.SESSION_ID}/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${CONFIG.TOKEN}`
            },
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showToast('موفق', 'فایل‌ها با موفقیت آپلود شدند', 'success');
            selectedFiles = [];
            renderFileList();
            document.getElementById('uploadBtn').disabled = true;
            bar.style.width = '100%';
            status.textContent = '✅ آپلود کامل شد!';
            loadDocuments();
        } else {
            showToast('خطا', result.message || 'خطا در آپلود', 'danger');
            status.textContent = '❌ خطا در آپلود';
        }
    } catch (error) {
        showToast('خطا', 'مشکل در ارتباط با سرور', 'danger');
        status.textContent = '❌ خطا در ارتباط';
    } finally {
        setTimeout(() => {
            progress.classList.add('d-none');
            bar.style.width = '0%';
            btn.disabled = false;
        }, 3000);
    }
}

// ===============================
// مدیریت اسناد
// ===============================
async function loadDocuments() {
    const container = document.getElementById('documentsList');
    
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/sessions/${CONFIG.SESSION_ID}/documents`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) throw new Error('Failed to load documents');
        
        const data = await response.json();
        const documents = data.documents || data.data || [];
        
        if (documents.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-file-pdf fa-3x mb-3 d-block"></i>
                    هنوز هیچ سندی آپلود نشده است
                </div>
            `;
            return;
        }
        
        container.innerHTML = `
            <div class="row">
                ${documents.map(doc => `
                    <div class="col-md-6 col-lg-4 mb-3">
                        <div class="card h-100 shadow-sm">
                            <div class="card-body">
                                <div class="d-flex align-items-start">
                                    <i class="fas fa-file-pdf text-danger fa-2x me-2"></i>
                                    <div>
                                        <h6 class="card-title mb-1">${doc.original_name || doc.name || 'بدون نام'}</h6>
                                        <small class="text-muted d-block">
                                            <span class="badge ${doc.processing_status === 'completed' ? 'bg-success' : 'bg-warning'}">
                                                ${doc.processing_status || 'pending'}
                                            </span>
                                        </small>
                                        <small class="text-muted d-block">
                                            📄 ${doc.page_count || 0} صفحه • 📊 ${formatFileSize(doc.file_size || 0)}
                                        </small>
                                        ${doc.processing_status === 'completed' ? `
                                            <small class="text-muted d-block">
                                                🧩 ${doc.metadata?.chunks || 0} چانک • 🕸️ ${doc.metadata?.entities || 0} موجودیت
                                            </small>
                                        ` : ''}
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer bg-transparent border-0">
                                <button class="btn btn-sm btn-outline-primary w-100" onclick="askAboutDocument(${doc.id})">
                                    <i class="fas fa-question me-1"></i> سوال از این سند
                                </button>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-1"></i> خطا در بارگذاری اسناد: ${error.message}
            </div>
        `;
    }
}

function askAboutDocument(docId) {
    document.querySelector('[data-tab="chat"]').click();
    setQuestion(`در مورد سند شماره ${docId} توضیح بده`);
}

document.getElementById('refreshDocsBtn').addEventListener('click', loadDocuments);

// ===============================
// پرسش و پاسخ
// ===============================
let currentQuestionId = null;
let pollInterval = null;

document.getElementById('askBtn').addEventListener('click', askQuestion);
document.getElementById('questionInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') askQuestion();
});

function setQuestion(text) {
    document.getElementById('questionInput').value = text;
}

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) {
        showToast('توجه', 'لطفاً یک سوال وارد کنید', 'warning');
        return;
    }
    
    input.value = '';
    addMessage('user', question);
    
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/sessions/${CONFIG.SESSION_ID}/search`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ content: question })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            addMessage('assistant', `❌ خطا: ${result.message || 'مشکل در پردازش سوال'}`);
            return;
        }
        
        currentQuestionId = result.question_id;
        addMessage('assistant', '⏳ در حال پردازش سوال شما...', true);
        
        // شروع polling برای دریافت نتیجه
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(checkAnswerStatus, CONFIG.POLL_INTERVAL);
        
    } catch (error) {
        addMessage('assistant', `❌ خطا در ارتباط با سرور: ${error.message}`);
    }
}

async function checkAnswerStatus() {
    if (!currentQuestionId) return;
    
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/questions/${currentQuestionId}/status`, {
            headers: getAuthHeaders()
        });
        
        const data = await response.json();
        
        if (data.is_completed) {
            clearInterval(pollInterval);
            pollInterval = null;
            await getAnswerResult();
        }
        
    } catch (error) {
        console.error('Status check error:', error);
    }
}

async function getAnswerResult() {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/questions/${currentQuestionId}/result`, {
            headers: getAuthHeaders()
        });
        
        const data = await response.json();
        
        // Remove loading message
        const loadingMsg = document.querySelector('.message.assistant .bubble .spinner-border');
        if (loadingMsg) {
            const msgEl = loadingMsg.closest('.message');
            if (msgEl) msgEl.remove();
        }
        
        if (data.answer) {
            let sourcesHtml = '';
            if (data.sources && data.sources.length > 0) {
                sourcesHtml = `
                    <div class="sources" onclick="showSources(${JSON.stringify(data.sources).replace(/"/g, '&quot;')})">
                        📚 ${data.sources.length} منبع
                    </div>
                `;
            }
            
            addMessage('assistant', data.answer + sourcesHtml);
            loadHistory();
        } else if (data.status === 'failed') {
            addMessage('assistant', `❌ خطا: ${data.error || 'پاسخی دریافت نشد'}`);
        } else {
            addMessage('assistant', '❌ پاسخ دریافت نشد');
        }
        
        currentQuestionId = null;
        
    } catch (error) {
        console.error('Get result error:', error);
    }
}

function showSources(sources) {
    let html = '<div class="p-2">📚 <strong>منابع:</strong><ul class="mb-0 mt-1">';
    sources.forEach(s => {
        html += `<li class="text-muted small">سند ${s.document_id}: ${s.search_type || 'unknown'}</li>`;
    });
    html += '</ul></div>';
    showToast('منابع', html, 'info');
}

function addMessage(role, content, isLoading = false) {
    const container = document.getElementById('chatMessages');
    const emptyChat = document.getElementById('emptyChat');
    if (emptyChat) emptyChat.style.display = 'none';
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    let contentHtml = content;
    if (isLoading) {
        contentHtml = `<div class="spinner-border text-primary spinner-border-sm me-2"></div> در حال پردازش...`;
    } else {
        contentHtml = content.replace(/\n/g, '<br>');
    }
    
    msgDiv.innerHTML = `<div class="bubble">${contentHtml}</div>`;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

// ===============================
// تاریخچه سوالات
// ===============================
async function loadHistory() {
    const container = document.getElementById('historyList');
    
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/sessions/${CONFIG.SESSION_ID}/questions`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) throw new Error('Failed to load history');
        
        const data = await response.json();
        const questions = data.questions || data.data || [];
        
        if (questions.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="fas fa-history fa-3x mb-3 d-block"></i>
                    هنوز هیچ سوالی پرسیده نشده است
                </div>
            `;
            return;
        }
        
        container.innerHTML = questions.map(q => `
            <div class="card shadow-sm mb-2">
                <div class="card-body py-2">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <div class="fw-bold">${q.content}</div>
                            ${q.status === 'completed' ? `
                                <small class="text-muted">${q.answer ? q.answer.substring(0, 150) + '...' : 'پاسخ موجود'}</small>
                            ` : `
                                <span class="badge status-${q.status}">${q.status}</span>
                            `}
                        </div>
                        <small class="text-muted">${new Date(q.created_at).toLocaleString('fa-IR')}</small>
                    </div>
                    ${q.status === 'completed' ? `
                        <button class="btn btn-sm btn-outline-primary mt-1" onclick="viewAnswer(${q.id})">
                            <i class="fas fa-eye me-1"></i> مشاهده پاسخ
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        container.innerHTML = `
            <div class="alert alert-danger">خطا در بارگذاری تاریخچه: ${error.message}</div>
        `;
    }
}

async function viewAnswer(questionId) {
    try {
        const response = await fetch(`${CONFIG.BASE_URL}/questions/${questionId}/result`, {
            headers: getAuthHeaders()
        });
        
        const data = await response.json();
        
        document.querySelector('[data-tab="chat"]').click();
        addMessage('assistant', data.answer || 'پاسخی یافت نشد');
        
        if (data.sources && data.sources.length > 0) {
            let sourcesHtml = '<div class="sources" onclick="alert(\'' + JSON.stringify(data.sources) + '\')">📚 ' + data.sources.length + ' منبع</div>';
            const lastMsg = document.querySelector('.message.assistant:last-child .bubble');
            if (lastMsg) {
                lastMsg.innerHTML += sourcesHtml;
            }
        }
        
    } catch (error) {
        showToast('خطا', 'مشکل در دریافت پاسخ', 'danger');
    }
}

document.getElementById('clearHistoryBtn').addEventListener('click', () => {
    if (confirm('آیا از پاک کردن تاریخچه مطمئن هستید؟')) {
        // اینجا باید API پاک کردن تاریخچه را فراخوانی کنید
        showToast('موفق', 'تاریخچه پاک شد', 'success');
        loadHistory();
    }
});

// ===============================
// لاگین / خروج
// ===============================
function setUserInfo(name, email) {
    document.getElementById('userName').textContent = name;
    document.getElementById('userEmail').textContent = email;
}

document.getElementById('logoutBtn').addEventListener('click', () => {
    localStorage.removeItem('api_token');
    showToast('خروج', 'شما از حساب خود خارج شدید', 'warning');
    setTimeout(() => {
        window.location.href = '/login.html';
    }, 1000);
});

// ===============================
// مقداردهی اولیه
// ===============================
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('api_token');
    if (!token) {
        showToast('توجه', 'لطفاً ابتدا وارد شوید', 'warning');
        // ریدایرکت به صفحه لاگین
        // window.location.href = '/login.html';
    } else {
        CONFIG.TOKEN = token;
        loadDocuments();
        loadHistory();
    }
});

// ===============================
// مدیریت سشن
// ===============================
// اگر می‌خواهید سشن را تغییر دهید، این تابع را صدا بزنید
function setSession(sessionId) {
    CONFIG.SESSION_ID = sessionId;
    document.getElementById('sessionBadge').textContent = `سشن: ${sessionId}`;
    loadDocuments();
    loadHistory();
}

// ===============================
// تابع برای لاگین (برای استفاده در صفحه لاگین)
// ===============================
function setToken(token) {
    localStorage.setItem('api_token', token);
    CONFIG.TOKEN = token;
}