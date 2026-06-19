document.addEventListener('DOMContentLoaded', () => {
    // Sliders setup
    const setupSlider = (sliderId, displayId) => {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(displayId);
        
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                display.textContent = e.target.value;
            });
        }
    };

    setupSlider('confidence', 'conf-val');
    setupSlider('fps_sample', 'fps-val');
    setupSlider('resize_factor', 'resize-val');

    // Model ID Toggle
    const modelTypeSelect = document.getElementById('model_type');
    const modelIdGroup = document.getElementById('model_id_group');
    const modelIdSelect = document.getElementById('model_id_select');

    const syncModelFields = () => {
        if (modelTypeSelect.value === 'yolo') {
            modelIdSelect.disabled = false;
            modelIdGroup.classList.remove('field-disabled');
        } else {
            modelIdSelect.disabled = true;
            modelIdGroup.classList.add('field-disabled');
        }
    };
    modelTypeSelect.addEventListener('change', syncModelFields);
    syncModelFields(); // Run on startup


    // Drag and drop setup
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-file');
    const fileNameDisplay = document.getElementById('file-name-display');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            updateFileName();
        }
    }

    fileInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
        } else {
            fileNameDisplay.textContent = '';
        }
    }

    // ─── Sidebar ───
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebarOpenBtn = document.getElementById('sidebar-open-btn');
    const historyList = document.getElementById('history-list');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');

    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.add('collapsed');
        sidebarOpenBtn.classList.remove('hidden');
    });

    sidebarOpenBtn.addEventListener('click', () => {
        sidebar.classList.remove('collapsed');
        sidebarOpenBtn.classList.add('hidden');
    });

    // New Analysis button in sidebar — resets to upload form
    newAnalysisBtn.addEventListener('click', () => {
        resetApp();
    });

    // ─── Advanced Settings Toggle ───
    const advancedToggle = document.getElementById('advanced-settings-toggle');
    const advancedContent = document.getElementById('advanced-settings-content');

    advancedToggle.addEventListener('click', () => {
        advancedToggle.classList.toggle('open');
        advancedContent.classList.toggle('open');
    });

    // Currently active history folder (for highlighting)
    let activeHistoryFolder = null;

    function loadHistory() {
        fetch('/api/history')
            .then(res => res.json())
            .then(runs => {
                renderHistoryList(runs);
            })
            .catch(err => {
                console.error('Failed to load history:', err);
            });
    }

    function renderHistoryList(runs) {
        if (!runs || runs.length === 0) {
            historyList.innerHTML = `
                <div class="sidebar-empty">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.3">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                    <p>No analyses yet</p>
                </div>
            `;
            return;
        }

        historyList.innerHTML = '';
        runs.forEach(run => {
            const item = document.createElement('div');
            item.className = 'history-item';
            if (run.folder === activeHistoryFolder) {
                item.classList.add('active');
            }
            item.dataset.folder = run.folder;

            const modelBadge = run.model_info
                ? `<span class="meta-badge">${run.model_info.model_type}</span>`
                : '';

            item.innerHTML = `
                <div class="history-item-name" title="${run.video_name || run.folder}">${run.video_name || run.folder}</div>
                <div class="history-item-meta">
                    <span>${run.run_date || '—'}</span>
                    ${modelBadge}
                </div>
            `;

            item.addEventListener('click', () => {
                loadHistoryDetail(run);
                // Highlight active item
                activeHistoryFolder = run.folder;
                historyList.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
                item.classList.add('active');
            });

            historyList.appendChild(item);
        });
    }

    function loadHistoryDetail(run) {
        // Show results from a history entry
        const results = run.files;
        const modelInfo = run.model_info;
        const analysisSettings = run.analysis_settings;

        // Translate files structure to the same format showResults expects
        const normalizedResults = {
            folder: `/output/${run.folder}`,
            video: results.video,
            csv: results.csv,
            json: results.json,
            qa_json_files: results.qa_json_files || [],
        };

        showResults(normalizedResults, modelInfo, run.video_name || run.folder, analysisSettings);
    }

    // Form submission
    const form = document.getElementById('upload-form');
    const mainPanel = document.getElementById('main-panel');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    const resultsPanel = document.getElementById('results-panel');
    const errorPanel = document.getElementById('error-panel');
    const errorMessage = document.getElementById('error-message');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please select a video file first.');
            return;
        }

        const formData = new FormData(form);
        
        // Convert checkboxes to explicit boolean values
        if (!formData.has('save_sampled_only')) {
            formData.append('save_sampled_only', 'false');
        } else {
            formData.set('save_sampled_only', 'true');
        }

        formData.set('generate_video', document.getElementById('generate_video').checked ? 'true' : 'false');
        formData.set('generate_csv', document.getElementById('generate_csv').checked ? 'true' : 'false');
        formData.set('generate_json', document.getElementById('generate_json').checked ? 'true' : 'false');

        // Processing options
        formData.set('remove_audio', document.getElementById('remove_audio').checked ? 'true' : 'false');
        formData.set('mask_persons', document.getElementById('mask_persons').checked ? 'true' : 'false');

        // Compile QA Categories — generate_qa is true if any category is checked
        const activeQaCategories = [];
        if (document.getElementById('qa_counting').checked) activeQaCategories.push('counting');
        if (document.getElementById('qa_negative').checked) activeQaCategories.push('negative');
        if (document.getElementById('qa_ambiguity').checked) activeQaCategories.push('ambiguity');
        if (document.getElementById('qa_day_night').checked) activeQaCategories.push('day_night');
        formData.set('generate_qa', activeQaCategories.length > 0 ? 'true' : 'false');
        formData.set('qa_categories', activeQaCategories.join(','));

        // Show loading state
        mainPanel.classList.add('hidden');
        loadingOverlay.classList.remove('hidden');
        resultsPanel.classList.add('hidden');
        errorPanel.classList.add('hidden');

        try {
            loadingStatus.textContent = 'Uploading...';
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            const taskId = data.task_id;

            pollStatus(taskId);

        } catch (error) {
            showError('Failed to start analysis: ' + error.message);
        }
    });

    // Polling function
    async function pollStatus(taskId) {
        try {
            const response = await fetch(`/api/status/${taskId}`);
            
            if (!response.ok) {
                throw new Error('Status check failed');
            }
            
            const data = await response.json();
            
            if (data.status === 'completed') {
                showResults(data.results, data.model_info, undefined, data.analysis_settings);
                // Refresh the sidebar history to include the new run
                loadHistory();
            } else if (data.status === 'error') {
                showError(data.error || 'Unknown error occurred during analysis');
            } else {
                const progressContainer = document.getElementById('progress-container');
                const progressFill = document.getElementById('progress-fill');
                const loadingSpinner = document.getElementById('loading-spinner');
                
                // Update loading text based on status
                if (data.status === 'loading_model') {
                    loadingStatus.textContent = 'Loading AI Model...';
                    progressContainer.classList.add('hidden');
                    loadingSpinner.style.display = 'block';
                } else if (data.status === 'analyzing') {
                    loadingStatus.textContent = `Analyzing Video Frames (${data.progress || 0}%)`;
                    progressContainer.classList.remove('hidden');
                    loadingSpinner.style.display = 'none';
                    progressFill.style.width = `${data.progress || 0}%`;
                } else {
                    loadingStatus.textContent = 'Processing...';
                    progressContainer.classList.add('hidden');
                    loadingSpinner.style.display = 'block';
                }
                
                // Continue polling
                setTimeout(() => pollStatus(taskId), 2000);
            }
        } catch (error) {
            showError('Lost connection to server: ' + error.message);
        }
    }

    // Helper: build a /api/download URL from an output-relative path
    function downloadUrl(path) {
        return `/api/download?path=${encodeURIComponent(path)}`;
    }

    // Helper: build a streaming video URL from an output-relative path
    function streamVideoUrl(path) {
        return `/api/stream-video?path=${encodeURIComponent(path)}`;
    }

    // Display functions
    function showResults(results, modelInfo, displayName, analysisSettings) {
        loadingOverlay.classList.add('hidden');
        mainPanel.classList.add('hidden');
        resultsPanel.classList.remove('hidden');
        errorPanel.classList.add('hidden');

        // Set filename in the result header
        const filename = displayName || (fileInput.files.length > 0 ? fileInput.files[0].name : 'Analysis Complete');
        document.getElementById('result-filename').textContent = filename;

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            const hasModel = modelInfo && (modelInfo.model_type || modelInfo.model_name);
            const hasSettings = analysisSettings && Object.values(analysisSettings).some(v => v != null);
            if (hasModel || hasSettings) {
                metaDisplay.style.display = 'grid';
                let html = '';
                if (hasModel) {
                    html += `<span>Detector: <strong>${modelInfo.model_type || '—'}</strong></span>`;
                    html += `<span>Model: <strong>${modelInfo.model_name || '—'}</strong></span>`;
                }
                if (hasSettings) {
                    if (analysisSettings.resolution)
                        html += `<span>Resolution: <strong>${analysisSettings.resolution}</strong></span>`;
                    if (analysisSettings.total_frames != null)
                        html += `<span>Frames: <strong>${analysisSettings.total_frames.toLocaleString()}</strong></span>`;
                    if (analysisSettings.fps != null)
                        html += `<span>FPS: <strong>${analysisSettings.fps}</strong></span>`;
                    if (analysisSettings.duration_seconds != null) {
                        const dur = analysisSettings.duration_seconds;
                        const mins = Math.floor(dur / 60);
                        const secs = Math.round(dur % 60);
                        html += `<span>Duration: <strong>${mins}:${String(secs).padStart(2,'0')}</strong></span>`;
                    }
                    if (analysisSettings.fps_sample != null)
                        html += `<span>FPS Sampling: <strong>${analysisSettings.fps_sample}</strong></span>`;
                    if (analysisSettings.confidence_threshold != null)
                        html += `<span>Confidence: <strong>${analysisSettings.confidence_threshold}</strong></span>`;
                }
                metaDisplay.innerHTML = html;
            } else {
                metaDisplay.style.display = 'none';
            }
        }
        
        const videoPlayer = document.getElementById('result-video');
        const downloadCsv = document.getElementById('download-csv');
        const downloadJson = document.getElementById('download-json');
        const downloadVideo = document.getElementById('download-video');
        const analysisSection = document.getElementById('analysis-downloads').closest('.output-section');

        let hasAnalysisFiles = false;
        if (results.video) {
            hasAnalysisFiles = true;
            const videoContainer = videoPlayer.closest('.video-container');

            // Reset any previous error state
            const existingError = videoContainer.querySelector('.video-error-msg');
            if (existingError) existingError.remove();
            videoPlayer.style.display = 'block';

            videoPlayer.src = streamVideoUrl(results.video);
            videoPlayer.load();
            videoContainer.style.display = 'block';
            downloadVideo.href = downloadUrl(results.video);
            downloadVideo.removeAttribute('download');
            downloadVideo.style.display = 'flex';

            // Handle video decode errors (e.g., mp4v codec not supported by browser)
            videoPlayer.onerror = () => {
                videoPlayer.style.display = 'none';
                if (!videoContainer.querySelector('.video-error-msg')) {
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'video-error-msg';
                    errorMsg.innerHTML = `
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.5">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="15" y1="9" x2="9" y2="15"></line>
                            <line x1="9" y1="9" x2="15" y2="15"></line>
                        </svg>
                        <p>This video uses a codec (mp4v) that your browser cannot play.</p>
                        <p class="video-error-hint">Use the download button below to play it locally, or re-analyze with the <strong>avc1 (H.264)</strong> codec for browser playback.</p>
                    `;
                    videoContainer.appendChild(errorMsg);
                }
            };
        } else {
            videoPlayer.style.display = 'none';
            videoPlayer.closest('.video-container').style.display = 'none';
            downloadVideo.style.display = 'none';
        }
        
        if (results.csv) {
            hasAnalysisFiles = true;
            downloadCsv.href = downloadUrl(results.csv);
            downloadCsv.removeAttribute('download');
            downloadCsv.style.display = 'flex';
        } else {
            downloadCsv.style.display = 'none';
        }
        
        if (results.json) {
            hasAnalysisFiles = true;
            downloadJson.href = downloadUrl(results.json);
            downloadJson.removeAttribute('download');
            downloadJson.style.display = 'flex';
        } else {
            downloadJson.style.display = 'none';
        }

        // Show or hide the Analysis Outputs section
        if (hasAnalysisFiles) {
            analysisSection.classList.remove('hidden');
        } else {
            analysisSection.classList.add('hidden');
        }


        // Render per-category QA download buttons
        const qaSection = document.getElementById('qa-output-section');
        const qaContainer = document.getElementById('qa-downloads-container');
        qaContainer.innerHTML = '';
        const qaFiles = results.qa_json_files || (results.qa_json ? [results.qa_json] : []);
        if (qaFiles.length > 0) {
            qaSection.classList.remove('hidden');
            qaFiles.forEach(filePath => {
                const match = filePath.match(/_qa_([^/]+)\.json$/);
                const label = match
                    ? 'Download QA: ' + match[1].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                    : 'Download QA Pairs';
                const svgIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>`;
                const btn = document.createElement('a');
                btn.href = downloadUrl(filePath);
                btn.className = 'secondary-btn';
                btn.innerHTML = svgIcon + ' ' + label;
                qaContainer.appendChild(btn);
            });
        } else {
            qaSection.classList.add('hidden');
        }
    }

    function showError(msg) {
        loadingOverlay.classList.add('hidden');
        errorPanel.classList.remove('hidden');
        errorMessage.textContent = msg;
    }

    // Reset buttons
    document.getElementById('reset-btn').addEventListener('click', resetApp);
    document.getElementById('error-reset-btn').addEventListener('click', resetApp);

    function resetApp() {
        form.reset();
        fileNameDisplay.textContent = '';
        document.getElementById('conf-val').textContent = '0.7';
        document.getElementById('fps-val').textContent = '1.0';
        document.getElementById('resize-val').textContent = '1.0';
        
        // Sync conditional UI states
        syncModelFields();

        // Reset progress
        document.getElementById('progress-container').classList.add('hidden');
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('loading-spinner').style.display = 'block';

        // Hide result sections
        document.getElementById('qa-downloads-container').innerHTML = '';
        document.getElementById('qa-output-section').classList.add('hidden');
        document.getElementById('analysis-downloads').closest('.output-section').classList.add('hidden');

        const metaDisplay = document.getElementById('meta-info-display');
        if (metaDisplay) {
            metaDisplay.innerHTML = '';
            metaDisplay.style.display = 'none';
        }

        // Deselect history item
        activeHistoryFolder = null;
        historyList.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));

        errorPanel.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        mainPanel.classList.remove('hidden');
    }

    // Preload results if video query parameter is passed
    const urlParams = new URLSearchParams(window.location.search);
    const videoParam = urlParams.get('video');
    if (videoParam) {
        fetch(`/api/results?video=${encodeURIComponent(videoParam)}`)
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('No preloaded results found');
            })
            .then(data => {
                if (data.status === 'completed') {
                    showResults(data.results, data.model_info, videoParam, data.analysis_settings);
                    fileNameDisplay.textContent = videoParam;
                    mainPanel.classList.add('hidden');
                }
            })
            .catch(err => {
                console.log('No preloaded results found:', err.message);
            });
    }

    // Load history on startup
    loadHistory();
});
