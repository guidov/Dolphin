/**
 * Dolphin PDF Parser - Frontend JavaScript
 * ==========================================
 * Handles file uploads, API communication, and UI interactions
 */

// ============================================
// State Management
// ============================================

let currentResult = null;
let currentTab = 'markdown';

// ============================================
// DOM Elements
// ============================================

const elements = {
    uploadSection: document.getElementById('upload-section'),
    processingSection: document.getElementById('processing-section'),
    resultsSection: document.getElementById('results-section'),
    errorSection: document.getElementById('error-section'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    processingStatus: document.getElementById('processing-status'),
    progressFill: document.getElementById('progress-fill'),
    resultsContent: document.getElementById('results-content'),
    pageInfo: document.getElementById('page-info'),
    errorMessage: document.getElementById('error-message'),
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    setupFileInput();
    initParticles();
});

// ============================================
// Particle Background Effect
// ============================================

function initParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    // Create floating particles
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 2}px;
            height: ${Math.random() * 4 + 2}px;
            background: rgba(99, 102, 241, ${Math.random() * 0.3 + 0.1});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: particle-float ${Math.random() * 10 + 10}s ease-in-out infinite;
            animation-delay: ${Math.random() * 5}s;
        `;
        container.appendChild(particle);
    }

    // Add particle animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes particle-float {
            0%, 100% { 
                transform: translateY(0) translateX(0); 
                opacity: 0.3;
            }
            25% {
                transform: translateY(-30px) translateX(10px);
                opacity: 0.6;
            }
            50% { 
                transform: translateY(-20px) translateX(-10px); 
                opacity: 0.4;
            }
            75% {
                transform: translateY(-40px) translateX(5px);
                opacity: 0.5;
            }
        }
    `;
    document.head.appendChild(style);
}

// ============================================
// Drag and Drop Setup
// ============================================

function setupDragAndDrop() {
    const { dropZone } = elements;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when dragging over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Click to open file browser
    dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            elements.fileInput.click();
        }
    });
}

// ============================================
// File Input Setup
// ============================================

function setupFileInput() {
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
}

// ============================================
// File Handling
// ============================================

async function handleFile(file) {
    // Validate file type
    const allowedExtensions = ['.pdf', '.png', '.jpg', '.jpeg'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedExtensions.includes(ext)) {
        showError(`Unsupported file type "${ext}". Please upload PDF, PNG, or JPG files.`);
        return;
    }

    // Validate file size (20MB max)
    const maxSize = 20 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(`File too large. Maximum size is 20MB.`);
        return;
    }

    // Show processing UI
    showSection('processing');
    updateStatus('Uploading file...');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);

        // Start processing
        updateStatus('Processing document with Dolphin AI...');

        const response = await fetch('/api/convert', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            let errorDetail = 'Processing failed';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) {
                // Use default error message
            }
            throw new Error(errorDetail);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Processing failed');
        }

        // Store result and display
        currentResult = result;
        currentResult.filename = file.name;
        showResults(result);

    } catch (error) {
        console.error('Processing error:', error);
        showError(error.message || 'An unexpected error occurred. Please try again.');
    }
}

// ============================================
// Status Updates
// ============================================

function updateStatus(status) {
    if (elements.processingStatus) {
        elements.processingStatus.textContent = status;
    }
}

// ============================================
// Section Navigation
// ============================================

function showSection(section) {
    // Hide all sections
    elements.uploadSection?.classList.add('hidden');
    elements.processingSection?.classList.add('hidden');
    elements.resultsSection?.classList.add('hidden');
    elements.errorSection?.classList.add('hidden');

    // Show requested section
    switch (section) {
        case 'upload':
            elements.uploadSection?.classList.remove('hidden');
            break;
        case 'processing':
            elements.processingSection?.classList.remove('hidden');
            break;
        case 'results':
            elements.resultsSection?.classList.remove('hidden');
            break;
        case 'error':
            elements.errorSection?.classList.remove('hidden');
            break;
    }
}

// ============================================
// Results Display
// ============================================

function showResults(result) {
    const content = result.markdown || result.content || 'No content extracted';

    // Render content based on current tab
    renderContent(content, currentTab);

    // Show page info
    if (result.total_pages) {
        elements.pageInfo.textContent = `✨ Extracted from ${result.total_pages} page(s) · ${result.filename || 'document'}`;
    } else {
        elements.pageInfo.textContent = `✨ Extracted from ${result.filename || 'document'}`;
    }

    showSection('results');
}

function renderContent(content, viewType) {
    if (viewType === 'raw') {
        // Raw text view
        elements.resultsContent.classList.add('raw-view');
        elements.resultsContent.textContent = content;
    } else {
        // Markdown view with basic rendering
        elements.resultsContent.classList.remove('raw-view');
        elements.resultsContent.innerHTML = renderMarkdown(content);
    }
}

// ============================================
// Markdown Rendering
// ============================================

function renderMarkdown(text) {
    if (!text) return '';

    let html = text;

    // Escape HTML first
    html = html
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks (before inline code)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers (must check longer patterns first)
    html = html.replace(/^#### (.*$)/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/_(.+?)_/g, '<em>$1</em>');

    // Blockquotes
    html = html.replace(/^&gt; (.*$)/gm, '<blockquote>$1</blockquote>');

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');
    html = html.replace(/^\*\*\*$/gm, '<hr>');

    // Lists (basic)
    html = html.replace(/^[\-\*] (.*$)/gm, '<li>$1</li>');
    html = html.replace(/^(\d+)\. (.*$)/gm, '<li>$2</li>');

    // Wrap consecutive li elements in ul
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
        return '<ul>' + match + '</ul>';
    });

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Paragraphs - wrap lines that aren't already wrapped
    const lines = html.split('\n');
    html = lines.map(line => {
        const trimmed = line.trim();
        if (!trimmed) return '';
        if (trimmed.startsWith('<h') ||
            trimmed.startsWith('<ul') ||
            trimmed.startsWith('<ol') ||
            trimmed.startsWith('<li') ||
            trimmed.startsWith('<pre') ||
            trimmed.startsWith('<blockquote') ||
            trimmed.startsWith('<hr')) {
            return line;
        }
        return line;
    }).join('<br>');

    // Clean up extra br tags
    html = html.replace(/<br><br>/g, '</p><p>');
    html = html.replace(/(<h[1-6]>)/g, '</p>$1');
    html = html.replace(/(<\/h[1-6]>)/g, '$1<p>');

    return html;
}

// ============================================
// Tab Switching
// ============================================

function switchTab(tab) {
    currentTab = tab;

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tab) {
            btn.classList.add('active');
        }
    });

    // Re-render content
    if (currentResult) {
        const content = currentResult.markdown || currentResult.content || '';
        renderContent(content, tab);
    }
}

// ============================================
// Error Handling
// ============================================

function showError(message) {
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
    }
    showSection('error');
}

// ============================================
// App Reset
// ============================================

function resetApp() {
    currentResult = null;
    currentTab = 'markdown';

    // Reset file input
    if (elements.fileInput) {
        elements.fileInput.value = '';
    }

    // Reset tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === 'markdown') {
            btn.classList.add('active');
        }
    });

    showSection('upload');
}

// ============================================
// Clipboard & Download Functions
// ============================================

async function copyToClipboard() {
    if (!currentResult) return;

    const text = currentResult.markdown || currentResult.content || '';

    try {
        await navigator.clipboard.writeText(text);
        showToast('✅ Copied to clipboard!');
    } catch (err) {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast('✅ Copied to clipboard!');
    }
}

function downloadMarkdown() {
    if (!currentResult) return;

    const text = currentResult.markdown || currentResult.content || '';
    const filename = (currentResult.filename || 'document').replace(/\.[^/.]+$/, '') + '.md';

    downloadFile(text, filename, 'text/markdown');
    showToast('⬇️ Downloading markdown file...');
}

function downloadJSON() {
    if (!currentResult) return;

    const json = JSON.stringify(currentResult, null, 2);
    const filename = (currentResult.filename || 'document').replace(/\.[^/.]+$/, '') + '.json';

    downloadFile(json, filename, 'application/json');
    showToast('⬇️ Downloading JSON file...');
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message) {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());

    // Create new toast
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Remove after animation
    setTimeout(() => {
        toast.remove();
    }, 2500);
}
