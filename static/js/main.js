// Global State
let passages = [];
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordStartTime = null;
let timerInterval = null;
let selectedFile = null;
let activeTab = 'record';

// Audio Visualizer Global Contexts
let audioCtx = null;
let analyser = null;
let sourceNode = null;
let animationFrameId = null;
const canvas = document.getElementById('waveform-canvas');
const canvasCtx = canvas.getContext('2d');

// Accent Flag Mapping
const flagEmojis = {
    'American English': '🇺🇸',
    'British English': '🇬🇧',
    'Indian English': '🇮🇳',
    'Nepali English': '🇳🇵',
    'Australian English': '🇦🇺',
    'Canadian English': '🇨🇦',
    'Russian English': '🇷🇺',
    'Arabic English': '🇸🇦',
    'Chinese English': '🇨🇳',
    'Japanese English': '🇯🇵'
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    // Setup Canvas dimensions
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Fetch and render reading passages
    fetchPassages();

    // Setup Drag and Drop event listeners
    setupDragAndDrop();

    // Build initial history container
    renderHistory();

    // Draw initial idle visualizer waveform
    drawIdleWaveform();
});

function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
}

// Fetch Reading Passages from API
async function fetchPassages() {
    try {
        const response = await fetch('/passages');
        passages = await response.json();

        const select = document.getElementById('passage-select');
        select.innerHTML = '';

        if (passages && passages.length > 0) {
            passages.forEach((p, idx) => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = p.theme;
                select.appendChild(option);
            });
            // Show first by default
            displayPassage(passages[0]);
        } else {
            select.innerHTML = '<option value="">No passages found</option>';
        }
    } catch (e) {
        console.error('Error fetching passages:', e);
        document.getElementById('passage-select').innerHTML = '<option value="">Failed to load passages</option>';
    }
}

function displayPassage(passage) {
    const display = document.getElementById('passage-display');
    display.textContent = passage.text;
}

function onPassageChange() {
    const select = document.getElementById('passage-select');
    const selectedPassage = passages.find(p => p.id === select.value);
    if (selectedPassage) {
        displayPassage(selectedPassage);
    }
}

function selectRandomPassage() {
    if (passages && passages.length > 0) {
        const randomIndex = Math.floor(Math.random() * passages.length);
        const select = document.getElementById('passage-select');
        select.selectedIndex = randomIndex;
        displayPassage(passages[randomIndex]);
    }
}

// Switch Tabs between Record and Upload
function switchTab(tab) {
    activeTab = tab;

    // Header styling updates
    document.getElementById('btn-tab-record').classList.toggle('active', tab === 'record');
    document.getElementById('btn-tab-upload').classList.toggle('active', tab === 'upload');

    // Tab display updates
    document.getElementById('tab-record').classList.toggle('active', tab === 'record');
    document.getElementById('tab-upload').classList.toggle('active', tab === 'upload');

    // If recording was active, abort it
    if (tab !== 'record' && isRecording) {
        abortRecording();
    }

    // Reset Canvas and show idle state
    if (tab === 'record') {
        resizeCanvas();
        drawIdleWaveform();
    }
}

// Drag & Drop Handling
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');

    // Trigger file selection dialog on click
    dropZone.addEventListener('click', () => {
        document.getElementById('audio-file-input').click();
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    }, false);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file formats
    const validExtensions = ['.wav', '.mp3', '.m4a'];
    const filename = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => filename.endsWith(ext));

    if (!isValid) {
        alert("Unsupported file format! Please upload .wav, .mp3, or .m4a audio.");
        return;
    }

    selectedFile = file;

    // Render file properties on screen
    document.getElementById('selected-file-name').textContent = file.name;
    document.getElementById('selected-file-size').textContent = formatBytes(file.size);

    // Toggle containers
    document.getElementById('drop-zone').style.display = 'none';
    document.getElementById('file-details-container').style.display = 'flex';
}

function clearSelectedFile() {
    selectedFile = null;
    document.getElementById('audio-file-input').value = '';
    document.getElementById('drop-zone').style.display = 'flex';
    document.getElementById('file-details-container').style.display = 'none';
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Media Recorder Mic Actions
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    audioChunks = [];

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Start Web Audio analyser nodes
        setupAudioVisualizer(stream);

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            // Send file to model predict endpoint
            await processPrediction(audioBlob);

            // Release microphone stream tracks
            stream.getTracks().forEach(track => track.stop());
        };

        // Begin captures
        mediaRecorder.start();
        isRecording = true;

        // UI toggle updates
        document.getElementById('btn-record-main').classList.add('recording');
        document.getElementById('main-record-btn-icon').className = 'fa-solid fa-square';
        document.getElementById('record-status-text').innerHTML = '<span style="color: var(--neon-rose)">RECORDING...</span> Click to stop and analyze';
        document.getElementById('visualizer-idle-text').style.display = 'none';

        // Timer updates
        recordStartTime = Date.now();
        document.getElementById('record-timer').textContent = "00:00";
        document.getElementById('record-timer').style.display = 'block';
        timerInterval = setInterval(updateTimer, 500);

    } catch (err) {
        console.error("Microphone access blocked:", err);
        alert("Microphone permission denied! Please allow mic access in your browser settings to proceed.");
    }
}

function updateTimer() {
    const elapsedMs = Date.now() - recordStartTime;
    const totalSecs = Math.floor(elapsedMs / 1000);
    const mins = String(Math.floor(totalSecs / 60)).padStart(2, '0');
    const secs = String(totalSecs % 60).padStart(2, '0');
    document.getElementById('record-timer').textContent = `${mins}:${secs}`;
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        // Reset recording components UI
        document.getElementById('btn-record-main').classList.remove('recording');
        document.getElementById('main-record-btn-icon').className = 'fa-solid fa-microphone';
        document.getElementById('record-status-text').textContent = 'Click the microphone to start reading';
        document.getElementById('record-timer').style.display = 'none';

        clearInterval(timerInterval);
        stopAudioVisualizer();
    }
}

function abortRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.onstop = null; // Unbind callback to avoid processing prediction
        mediaRecorder.stop();
        isRecording = false;

        document.getElementById('btn-record-main').classList.remove('recording');
        document.getElementById('main-record-btn-icon').className = 'fa-solid fa-microphone';
        document.getElementById('record-status-text').textContent = 'Click the microphone to start reading';
        document.getElementById('record-timer').style.display = 'none';

        clearInterval(timerInterval);
        stopAudioVisualizer();
        drawIdleWaveform();
    }
}

// Web Audio API Visualizer
function setupAudioVisualizer(stream) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;

    sourceNode = audioCtx.createMediaStreamSource(stream);
    sourceNode.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        if (!isRecording) return;

        animationFrameId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = '#060a12';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 1.5;
        let barHeight;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 1.7; // Scale height

            // Neon cyan to purple gradients
            const percent = i / bufferLength;
            const r = Math.floor(6 + percent * 162);
            const g = Math.floor(182 - percent * 127);
            const b = Math.floor(212 + percent * 45);

            canvasCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;

            // Mirror wave bars vertically in center
            const yPos = (canvas.height - barHeight) / 2;

            canvasCtx.fillRect(x, yPos, barWidth - 2, barHeight);
            x += barWidth;
        }
    }

    draw();
}

function stopAudioVisualizer() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    if (audioCtx) {
        audioCtx.close();
        audioCtx = null;
    }
    analyser = null;
    sourceNode = null;
}

function drawIdleWaveform() {
    canvasCtx.fillStyle = '#060a12';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    canvasCtx.lineWidth = 2;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, canvas.height / 2);
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();

    document.getElementById('visualizer-idle-text').style.display = 'flex';
}

// Upload file button trigger
function uploadAndPredict() {
    if (selectedFile) {
        processPrediction(selectedFile);
    }
}

// Prediction trigger
async function processPrediction(audioBlob) {
    // Show Loader UI
    document.getElementById('result-empty-state').style.display = 'none';
    document.getElementById('result-content-state').style.display = 'none';
    document.getElementById('result-loader-state').style.display = 'flex';
    document.getElementById('result-box').className = 'result-card glass-card loading';

    const formData = new FormData();
    formData.append('file', audioBlob, 'speech_audio.wav');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.status === 'success') {
            displayPredictionResults(result);
        } else {
            renderErrorState(result.message || "Failed to analyze speech metrics.");
        }

    } catch (e) {
        console.error("Prediction error:", e);
        renderErrorState("Server connection failed. Make sure the Flask service is running.");
    }
}

function displayPredictionResults(data) {
    document.getElementById('result-loader-state').style.display = 'none';
    document.getElementById('result-content-state').style.display = 'block';
    document.getElementById('result-box').className = 'result-card glass-card';

    // Set Main predictions
    const accent = data.accent;
    const confidence = data.confidence;

    document.getElementById('result-accent').textContent = accent;
    document.getElementById('result-confidence-text').textContent = `${confidence}%`;
    document.getElementById('result-flag').textContent = flagEmojis[accent] || '🌐';

    // Progress Circle animation (perimeter is 201px)
    const ring = document.getElementById('result-confidence-ring');
    const perimeter = 2 * Math.PI * 32;
    const offset = perimeter - (confidence / 100) * perimeter;
    ring.style.strokeDasharray = `${perimeter} ${perimeter}`;
    ring.style.strokeDashoffset = offset;

    // Generate Charts
    const chartContainer = document.getElementById('distribution-chart-container');
    chartContainer.innerHTML = '';

    const distribution = data.distribution;
    let rank = 1;

    for (const [classLabel, prob] of Object.entries(distribution)) {
        if (prob <= 0 && rank > 4) continue; // Show only top possibilities if they are non-zero

        const row = document.createElement('div');
        row.className = `bar-row ${classLabel === accent ? 'top-accent' : ''}`;

        row.innerHTML = `
            <span class="bar-label">${flagEmojis[classLabel] || '🌐'} ${classLabel}</span>
            <div class="bar-outer">
                <div class="bar-inner" style="width: ${prob}%"></div>
            </div>
            <span class="bar-val">${prob}%</span>
        `;

        chartContainer.appendChild(row);
        rank++;
    }

    // Add to prediction history
    addToHistory({
        accent: accent,
        confidence: confidence,
        flag: flagEmojis[accent] || '🌐',
        date: new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    });
}

function renderErrorState(errorMessage) {
    document.getElementById('result-loader-state').style.display = 'none';
    document.getElementById('result-content-state').style.display = 'none';
    document.getElementById('result-empty-state').style.display = 'flex';
    document.getElementById('result-box').className = 'result-card glass-card empty';

    // Temporarily replace content inside empty state to display error
    const label = document.querySelector('#result-empty-state h3');
    const desc = document.querySelector('#result-empty-state p');

    label.innerHTML = '<span style="color: var(--neon-rose)">Extraction Error</span>';
    desc.textContent = errorMessage;
}

// Prediction History Actions via LocalStorage
function getHistory() {
    const list = localStorage.getItem('aura_accent_history');
    return list ? JSON.parse(list) : [];
}

function addToHistory(item) {
    let history = getHistory();
    // Unique ID
    item.id = Date.now();
    // Prepend to show most recent first
    history.unshift(item);

    // Cap history limit to 10 entries
    if (history.length > 10) {
        history.pop();
    }

    localStorage.setItem('aura_accent_history', JSON.stringify(history));
    renderHistory();
}

function clearHistory() {
    localStorage.removeItem('aura_accent_history');
    renderHistory();
}

function renderHistory() {
    const container = document.getElementById('history-container');
    container.innerHTML = '';

    const history = getHistory();

    if (history.length === 0) {
        container.innerHTML = '<div class="history-empty">No history logs recorded yet</div>';
        return;
    }

    history.forEach(item => {
        const row = document.createElement('div');
        row.className = 'history-item';

        row.innerHTML = `
            <span class="history-flag">${item.flag}</span>
            <div class="history-details">
                <span class="history-accent">${item.accent}</span>
                <span class="history-date">${item.date}</span>
            </div>
            <span class="history-score">${item.confidence}%</span>
        `;

        container.appendChild(row);
    });
}
